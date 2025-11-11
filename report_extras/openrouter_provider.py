import logging
import os
import json
import time
import requests
from typing import Optional, List, Dict, Any, Union
# Import the base provider class
try:
    from .base_provider import BaseLLMProvider
except ImportError:
    from base_provider import BaseLLMProvider

# --- Logging Setup ---
app_logger = logging.getLogger('briefing_app')
llm_logger = logging.getLogger('briefing_llm')

class OpenRouterProvider(BaseLLMProvider):
    """
    Implements the BaseLLMProvider interface for OpenRouter API.
    """
    
    def __init__(self, config, prompts: Dict[str, str]):
        self.config = config
        self.prompts = prompts
        
        # Get OpenRouter API key from environment or config
        api_key = os.environ.get("OPENROUTER_API_KEY") or config.llm_settings.get('api_keys', {}).get('openrouter')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable or config setting not found.")
        
        self.api_key = api_key
        self.base_url = config.llm_settings.get('endpoints', {}).get('openrouter', 'https://openrouter.ai/api/v1/chat/completions')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Rate limiting delays based on config
        self.pro_delay_sec = 60.0 / config.pro_rpm
        self.flash_delay_sec = 60.0 / config.flash_rpm
        self.last_pro_call_time = 0
        self.last_flash_call_time = 0
        
        # Model names
        self.pro_model_name = config.pro_model_name or config.llm_settings.get('models', {}).get('openrouter', 'openrouter/andromeda-alpha')
        self.flash_model_name = config.flash_model_name or config.llm_settings.get('models', {}).get('openrouter', 'openrouter/andromeda-alpha')
        
        app_logger.info(f"Initialized OpenRouter with PRO model: {self.pro_model_name}")
        app_logger.info(f"Initialized OpenRouter with FLASH model: {self.flash_model_name}")

    def _apply_rate_limit(self, model_name: str):
        """Enforces a simple time-based rate limit based on model name."""
        if model_name == self.config.pro_model_name or self.pro_model_name in model_name:
            time_since_last_call = time.time() - self.last_pro_call_time
            wait_time = self.config.pro_delay_sec - time_since_last_call
            if wait_time > 0:
                app_logger.info(f"PRO rate limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            self.last_pro_call_time = time.time()
        else:  # flash models
            time_since_last_call = time.time() - self.last_flash_call_time
            wait_time = self.config.flash_delay_sec - time_since_last_call
            if wait_time > 0:
                app_logger.debug(f"FLASH rate limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            self.last_flash_call_time = time.time()

    def _call_llm_with_backoff(self, prompt: str, model_name: str) -> Optional[str]:
        """Makes an API call to OpenRouter with rate limiting and exponential backoff."""
        self._apply_rate_limit(model_name)
        
        # Prepare the payload for OpenRouter
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        for attempt in range(self.config.max_retries):
            try:
                llm_logger.debug(f"--- OPENROUTER REQUEST (Attempt {attempt + 1}) for {model_name} ---\n{prompt[:1000]}...")
                
                response = requests.post(self.base_url, headers=self.headers, json=payload)
                
                if response.status_code == 200:
                    response_data = response.json()
                    response_text = response_data['choices'][0]['message']['content'].strip()
                    llm_logger.debug(f"--- OPENROUTER RESPONSE from {model_name} ---\n{response_text}")
                    return response_text
                elif response.status_code == 429:  # Rate limited
                    if attempt < self.config.max_retries - 1:
                        delay = self.config.initial_backoff_sec * (2 ** attempt)
                        app_logger.warning(f"Rate limited by OpenRouter. Retrying in {delay} seconds... (Attempt {attempt + 1}/{self.config.max_retries})")
                        time.sleep(delay)
                    else:
                        app_logger.error(f"Fatal error: Rate limited after {self.config.max_retries} retries for {model_name}.", exc_info=True)
                        return None
                else:
                    app_logger.error(f"OpenRouter API returned status code {response.status_code}: {response.text}")
                    if attempt < self.config.max_retries - 1:
                        delay = self.config.initial_backoff_sec * (2 ** attempt)
                        time.sleep(delay)
                    else:
                        return None
            
            except Exception as e:
                app_logger.error(f"Fatal error during OpenRouter API call for {model_name}: {e}", exc_info=True)
                if attempt < self.config.max_retries - 1:
                    delay = self.config.initial_backoff_sec * (2 ** attempt)
                    time.sleep(delay)
                else:
                    return None
        
        return None

    def parse_meeting_batch(self, records: List[Dict[str, Any]], current_date: str) -> List[Dict[str, Any]]:
        """Uses FLASH model to parse meeting announcements into structured JSON."""
        if not records:
            return []
        
        app_logger.info(f"Starting meeting parsing using content-based batching for {len(records)} records...")
        
        # Use content-based batching for the flash model (for efficiency)
        content_batches = self.build_content_aware_batches(records, self.flash_model_name)
        app_logger.info(f"Split {len(records)} meetings into {len(content_batches)} content-aware batches")
        
        structured_meetings = []
        
        for batch_num, batch in enumerate(content_batches):
            app_logger.info(f"Processing meeting batch {batch_num + 1}/{len(content_batches)} with {len(batch)} records...")
            
            for i, record in enumerate(batch):
                app_logger.info(f"Parsing meeting {batch_num + 1}.{i+1}/{len(batch)}: {record['url']}")
                
                prompt_template = self.prompts['1_parse_meeting']
                prompt = prompt_template.replace("[CURRENT_DATE]", current_date)
                prompt = prompt.replace("[MEETING_URL]", record['url'])
                extracted_text = record.get('extracted_text', '')
                prompt = prompt.replace("[MEETING_TEXT]", extracted_text)
                
                response_text = self._call_llm_with_backoff(prompt, self.flash_model_name)
                parsed_json = self._parse_json_response(response_text, record['url'], is_list=False, source_content=extracted_text)
                
                if isinstance(parsed_json, dict):
                    structured_meetings.append(parsed_json)
        
        app_logger.info(f"Successfully parsed {len(structured_meetings)} meetings.")
        return structured_meetings

    def summarize_news_batch(self, records: List[Dict[str, Any]], current_date: str) -> List[Dict[str, Any]]:
        """Summarize news items using OpenRouter."""
        if not records:
            return []
            
        # Use flash model for initial summarization (efficient processing)
        app_logger.info(f"Starting initial news summarization using {self.flash_model_name} for {len(records)} records...")
        
        # Use content-based batching for the flash model
        content_batches = self.build_content_aware_batches(records, self.flash_model_name)
        app_logger.info(f"Split {len(records)} news items into {len(content_batches)} content-aware batches")
        
        initial_summaries = []
        processed_urls = set()

        for batch_num, batch in enumerate(content_batches):
            app_logger.info(f"Processing news batch {batch_num + 1}/{len(content_batches)} with {len(batch)} records...")
            
            for i, record in enumerate(batch):
                app_logger.info(f"Summarizing news {batch_num + 1}.{i+1}/{len(batch)}: {record['url']}")
                
                # FIX: Use 'or' to ensure text is an empty string if .get() returns None
                text = record.get('extracted_text') or ''
                original_text = text  # Keep original for fact checking
                
                # --- Apply Truncation Logic ---
                if len(text) > self.config.truncation_length:
                    text = (
                        f"[NOTE TO EDITOR: The following document was excerpted due to extreme length.]\n\n"
                        f"{text[:self.config.truncation_length]}..."
                    )
                
                prompt_template = self.prompts['2_summarize_news']
                prompt = prompt_template.replace("[CURRENT_DATE]", current_date)
                prompt = prompt.replace("[NEWS_URL]", record['url'])
                prompt = prompt.replace("[NEWS_TEXT]", text)

                response_text = self._call_llm_with_backoff(prompt, self.flash_model_name)
                parsed_json = self._parse_json_response(response_text, record['url'], is_list=False, source_content=original_text)
                
                if isinstance(parsed_json, dict):
                    initial_summaries.append(parsed_json)
                    processed_urls.add(record['url'])
        
        app_logger.info(f"Successfully generated initial summaries for {len(initial_summaries)} news items using {self.flash_model_name}.")
        
        # Then, for higher accuracy, refine key summaries using pro model if configured
        if hasattr(self.config, 'use_pro_for_refinement') and self.config.use_pro_for_refinement:
            app_logger.info(f"Refining top summaries using {self.pro_model_name}...")
            refined_summaries = self._refine_summaries_with_pro(initial_summaries, current_date)
            return refined_summaries
        else:
            return initial_summaries

    def _refine_summaries_with_pro(self, summaries: List[Dict[str, Any]], current_date: str) -> List[Dict[str, Any]]:
        """Use the Pro model to refine summaries if higher accuracy is needed."""
        refined_summaries = []
        
        for summary in summaries:
            url = summary.get('url', '')
            app_logger.info(f"Refining summary for {url} using {self.pro_model_name}...")
            
            # Get original text for higher-quality refinement
            # In a real implementation, we'd need to fetch the original text again
            # For now, we'll pass the existing summary through the pro model
            
            refinement_prompt = f"""
            You are a senior editor. Today's date is {current_date}.
            Below is a news summary that needs refinement for accuracy and clarity:
            
            {json.dumps(summary, indent=2)}
            
            Please return a refined version of this summary as a single JSON object with the same structure:
            {{
              "url": "...",
              "summary": "...",
              "headline": "...",
              "department": "...",
              "is_truncated": ...
            }}
            """
            
            response_text = self._call_llm_with_backoff(refinement_prompt, self.pro_model_name)
            # We can't do fact checking here since we don't have the original source
            refined_json = self._parse_json_response(response_text, url, is_list=False)
            
            if isinstance(refined_json, dict):
                refined_summaries.append(refined_json)
            else:
                # Keep original if refinement fails
                refined_summaries.append(summary)
        
        return refined_summaries

    def select_top_stories(self, all_summaries: List[Dict[str, Any]], current_date: str) -> List[str]:
        """Uses PRO model to select top stories."""
        app_logger.info("Starting top story selection (PRO model)...")
        
        if not all_summaries:
            app_logger.warning("No summaries provided to select top stories from.")
            return []

        # Create the text block for the prompt
        summary_list_text = "\n---\n".join(
            f"[SOURCE: {s['url']}]\nHeadline: {s['headline']}\nSummary: {s['summary']}"
            for s in all_summaries
        )
        
        prompt_template = self.prompts['3_select_top_stories']
        prompt = prompt_template.replace("[CURRENT_DATE]", current_date)
        prompt = prompt.replace("[NEWS_SUMMARIES_LIST]", summary_list_text)

        response_text = self._call_llm_with_backoff(prompt, self.pro_model_name)
        parsed_list = self._parse_json_response(response_text, "top_story_selection", is_list=True)
        
        if isinstance(parsed_list, list):
            app_logger.info(f"Successfully selected {len(parsed_list)} top stories.")
            return parsed_list
        
        app_logger.error("Failed to select top stories, returned data was not a list.")
        return []

    def format_final_briefing(self, target_date: str, current_date: str, meetings_json: List[Dict], all_summaries: List[Dict], top_story_urls: List[str]) -> Optional[str]:
        """Uses PRO model to assemble the final report."""
        app_logger.info("Starting final briefing formatting (PRO model)...")
        
        import datetime
        # 1. Format date for title
        try:
            date_obj = datetime.datetime.strptime(target_date, '%Y-%m-%d')
            formatted_date = date_obj.strftime('%B %d, %Y')
        except ValueError:
            formatted_date = target_date

        # 2. Format top story URLs
        top_story_list_text = json.dumps(top_story_urls, indent=2)
        
        # 3. Format all news summaries
        all_summaries_text = "\n---\n".join(json.dumps(s) for s in all_summaries)
        
        # 4. Format all meeting JSONs
        meetings_list_text = "\n---\n".join(json.dumps(m) for m in meetings_json)
        
        # 5. Build the final prompt
        prompt_template = self.prompts['4_format_final_briefing']
        prompt = prompt_template.replace("[CURRENT_DATE]", current_date)
        prompt = prompt.replace("[TARGET_DATE]", target_date)
        prompt = prompt.replace("[Month Day, YYYY]", formatted_date)
        prompt = prompt.replace("[TOP_STORY_URLS_LIST]", top_story_list_text)
        prompt = prompt.replace("[ALL_NEWS_SUMMARIES_LIST]", all_summaries_text)
        prompt = prompt.replace("[MEETING_JSON_LIST]", meetings_list_text)

        # 6. Call the model
        final_report = self._call_llm_with_backoff(prompt, self.pro_model_name)
        
        if final_report:
            app_logger.info("Successfully generated final briefing.")
        else:
            app_logger.error("Final briefing generation failed.")
            
        return final_report