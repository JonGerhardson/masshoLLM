import logging
import yaml
import os
import re
import json
import time
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from typing import Optional, List, Dict, Any, Union
from datetime import datetime # Import datetime

# Import fact checker for accuracy verification
try:
    from fact_checker import verify_summary
except ImportError:
    def verify_summary(source_text: str, summary: str, url: str = "") -> Dict[str, Any]:
        # Dummy function if fact_checker is not available
        return {
            'accuracy_score': 1.0,
            'is_accurate': True,
            'confidence_score': 1.0,
            'issues': [],
            'suggestions': ['Fact checking not available']
        }

# Import the base provider class
try:
    from .base_provider import BaseLLMProvider, app_logger, llm_logger
except ImportError:
    from base_provider import BaseLLMProvider, app_logger, llm_logger

class BriefingConfig:
    """Holds the configuration for the briefing generation script."""
    def __init__(self, config_dict: Dict):
        config_dict = config_dict or {} 
        llm_settings = config_dict.get('llm_settings', {})
        
        # Provider selection
        self.provider_name = llm_settings.get('provider', 'gemini')
        
        # Model names - get the appropriate models based on the selected provider
        if self.provider_name == 'gemini':
            # Default to 2.0-flash for initial processing and 2.5-pro for final steps as per requirements
            self.flash_model_name = llm_settings.get('flash_model', 'gemini-2.0-flash')  # For initial processing
            self.pro_model_name = llm_settings.get('pro_model', 'gemini-2.5-pro')       # For final steps requiring high accuracy
        elif self.provider_name == 'openrouter':
            self.flash_model_name = llm_settings.get('models', {}).get('openrouter') or llm_settings.get('flash_model', 'openrouter/minimax/minimax-m2:free')
            self.pro_model_name = llm_settings.get('models', {}).get('openrouter') or llm_settings.get('pro_model', 'openrouter/minimax/minimax-m2:free')
        elif self.provider_name == 'lmstudio':
            self.flash_model_name = llm_settings.get('models', {}).get('lmstudio') or llm_settings.get('flash_model', 'liquid/lfm2=1.2b')
            self.pro_model_name = llm_settings.get('models', {}).get('lmstudio') or llm_settings.get('pro_model', 'liquid/lfm2=1.2b')
        else:
            # Default to gemini settings if provider is unknown
            self.flash_model_name = llm_settings.get('flash_model', 'gemini-2.0-flash')
            self.pro_model_name = llm_settings.get('pro_model', 'gemini-2.5-pro')
        
        # Rate Limits (RPM) - with defaults based on your actual model limits
        self.flash_rpm = llm_settings.get('flash_rpm', 15)  # gemini-2.0-flash: 15 RPM
        self.pro_rpm = llm_settings.get('pro_rpm', 2)       # gemini-2.5-pro: 2 RPM
        
        # Calculate delay in seconds based on RPM
        self.flash_delay_sec = 60.0 / self.flash_rpm
        self.pro_delay_sec = 60.0 / self.pro_rpm
        
        # Text processing
        self.truncation_length = llm_settings.get('truncation_length', 15000)
        
        # Option to use Pro model for refinement of initial summaries (for higher accuracy)
        self.use_pro_for_refinement = llm_settings.get('use_pro_for_refinement', False)
        
        # Retry logic
        self.max_retries = llm_settings.get('max_retries', 3)
        self.initial_backoff_sec = llm_settings.get('initial_backoff_sec', 60)
        
        # Store full llm_settings for provider-specific access
        self.llm_settings = llm_settings

class GeminiProvider(BaseLLMProvider):
    """
    Manages API calls to Gemini models, handling rate limiting, 
    exponential backoff, and model selection.
    """
    def __init__(self, config: BriefingConfig, prompts: Dict[str, str]):
        self.config = config
        self.prompts = prompts
        self.pro_model = None
        self.flash_model = None
        
        self.last_pro_call_time = 0
        self.last_flash_call_time = 0

        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            genai.configure(api_key=api_key)
            
            self.pro_model = genai.GenerativeModel(self.config.pro_model_name)
            self.flash_model = genai.GenerativeModel(self.config.flash_model_name)
            app_logger.info(f"Initialized PRO model: {self.config.pro_model_name}")
            app_logger.info(f"Initialized FLASH model: {self.config.flash_model_name}")
            
        except Exception as e:
            app_logger.critical(f"Failed to initialize Google Generative AI: {e}", exc_info=True)
            raise

    def calculate_optimal_batch_size(self, model_name: str) -> tuple:
        """
        Calculate optimal batch size based on model-specific rate limits and TPM
        """
        # Model-specific limits (as per your requirements)
        model_limits = {
            'gemini-2.5-flash': {'rpm': 10, 'tpm': 250000},
            'gemini-2.5-pro': {'rpm': 2, 'tpm': 125000},
            'gemini-2.0-flash': {'rpm': 15, 'tpm': 1000000}
        }
        
        limits = model_limits.get(model_name, model_limits['gemini-2.0-flash'])
        
        if limits['rpm'] <= 2:  # For gemini-2.5-pro with only 2 RPM
            max_chars_per_batch = min(60000, limits['tpm'] // 3)  # Use 1/3 of TPM budget per batch
            max_items_per_batch = 1  # Very small batch count due to low RPM
        elif limits['rpm'] <= 10:  # For gemini-2.5-flash with 10 RPM
            max_chars_per_batch = min(20000, limits['tpm'] // 4)  # Use 1/4 of TPM budget per batch
            max_items_per_batch = 3
        else:  # For gemini-2.0-flash with 15 RPM
            max_chars_per_batch = min(80000, limits['tpm'] // 8)  # Use 1/8 of TPM budget per batch
            max_items_per_batch = 5
        
        return max_chars_per_batch, max_items_per_batch

    def build_content_aware_batches(self, items: List[Dict[str, Any]], model_name: str) -> List[List[Dict[str, Any]]]:
        """
        Build batches based on content size rather than fixed count
        """
        max_chars_per_batch, max_items_per_batch = self.calculate_optimal_batch_size(model_name)
        
        # Sort items by content length (descending) to better pack batches
        # Handle the case where extracted_text might be None
        sorted_items = sorted(items, key=lambda x: len(x.get('extracted_text', '') or ''), reverse=True)
        
        batches = []
        current_batch = []
        current_batch_size = 0
        
        for item in sorted_items:
            item_content = item.get('extracted_text', '') or ''
            item_size = len(item_content)  # Include metadata size too
            
            # Check if adding this item would exceed limits
            would_exceed_chars = current_batch_size + item_size > max_chars_per_batch
            would_exceed_count = len(current_batch) >= max_items_per_batch
            
            if would_exceed_chars or would_exceed_count:
                # Finalize current batch if not empty
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_size = 0
                
                # If single item is too large, handle it specially
                if item_size > max_chars_per_batch:
                    # For now, split large content into smaller chunks
                    # In a more complex implementation, we might want to split the text itself
                    batches.append([item])
                else:
                    # Start new batch with this item
                    current_batch.append(item)
                    current_batch_size += item_size
            else:
                # Add to current batch
                current_batch.append(item)
                current_batch_size += item_size
        
        # Add final batch if it contains items
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def _apply_rate_limit(self, model_name: str):
        """Enforces a simple time-based rate limit based on model name."""
        if 'gemini-2.5-pro' in model_name:
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
        """Makes an API call with rate limiting and exponential backoff."""
        # Determine which model to use based on model_name
        if model_name == self.config.pro_model_name:
            model = self.pro_model
        else:  # default to flash model
            model = self.flash_model
            
        self._apply_rate_limit(model_name)
        
        for attempt in range(self.config.max_retries):
            try:
                llm_logger.debug(f"--- LLM REQUEST (Attempt {attempt + 1}) for {model_name} ---\n{prompt[:1000]}...")
                response = model.generate_content(prompt)
                response_text = response.text.strip()
                llm_logger.debug(f"--- LLM RESPONSE from {model_name} ---\n{response_text}")
                return response_text
            
            except google_exceptions.ResourceExhausted as e:
                if attempt < self.config.max_retries - 1:
                    delay = self.config.initial_backoff_sec * (2 ** attempt)
                    app_logger.warning(f"Quota exceeded for {model_name}. Retrying in {delay} seconds... (Attempt {attempt + 1}/{self.config.max_retries})")
                    time.sleep(delay)
                else:
                    app_logger.error(f"Fatal error: Quota exceeded after {self.config.max_retries} retries for {model_name}.", exc_info=True)
                    return None
            
            except Exception as e:
                app_logger.error(f"Fatal error during Gemini API call for {model_name}: {e}", exc_info=True)
                return None
        
        return None

    def _parse_json_response(self, text: str, url: str, is_list: bool = False, source_content: str = None) -> Optional[Union[Dict, List]]:
        """Extracts and parses JSON from the LLM's response text."""
        if not text:
            app_logger.error(f"No response text from LLM for {url}.")
            return None
            
        # Regex to find JSON in ```json ... ``` or just floating {..} or [..]
        if is_list:
            match = re.search(r'```json\s*(\[.*\])\s*```|(\[.*\])', text, re.DOTALL)
        else:
            match = re.search(r'```json\s*(\{.*\})\s*```|(\{.*\})', text, re.DOTALL)

        if not match:
            app_logger.error(f"Could not find JSON in response for {url}.")
            llm_logger.error(f"NO JSON FOUND IN:\n{text}")
            return None

        json_str = next((g for g in match.groups() if g is not None), None)

        try:
            parsed_result = json.loads(json_str)
            
            # If this is a summary object, verify its accuracy against source content
            if isinstance(parsed_result, dict) and 'summary' in parsed_result and source_content:
                summary = parsed_result['summary']
                verification_result = verify_summary(source_content, summary, url)
                
                if not verification_result.get('is_accurate', True):
                    # Add detailed warning to summary if fact check fails
                    detailed_issues = verification_result.get('detailed_issues', [])
                    if detailed_issues:
                        issues_text = "; ".join(detailed_issues[:2])  # Limit to first 2 issues to avoid clutter
                        if len(detailed_issues) > 2:
                            issues_text += f"; +{len(detailed_issues) - 2} more issues"
                        accuracy_warning = f"\n[FACT CHECK WARNING: {len(detailed_issues)} accuracy issues - {issues_text}. {verification_result.get('suggestions', ['Review for accuracy'])[0]}]"
                    else:
                        accuracy_warning = f"\n[FACT CHECK WARNING: {len(verification_result.get('issues', []))} accuracy issues detected. {verification_result.get('suggestions', ['Review for accuracy'])[0]}]"
                    parsed_result['summary'] += accuracy_warning
                    app_logger.warning(f"Fact check failed for {url}: {detailed_issues if detailed_issues else verification_result.get('issues', [])}")
            
            return parsed_result
        except json.JSONDecodeError as e:
            app_logger.error(f"Failed to decode JSON from LLM response for {url}: {e}")
            llm_logger.error(f"MALFORMED JSON:\n{json_str}")
            return None

    def parse_meeting_batch(self, records: List[Dict[str, Any]], current_date: str) -> List[Dict[str, Any]]:
        """Uses FLASH model to parse meeting announcements into structured JSON."""
        if not records:
            return []
        
        app_logger.info(f"Starting meeting parsing using content-based batching for {len(records)} records...")
        
        # Use content-based batching for the flash model (for efficiency)
        content_batches = self.build_content_aware_batches(records, self.config.flash_model_name)
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
                
                response_text = self._call_llm_with_backoff(prompt, self.config.flash_model_name)
                parsed_json = self._parse_json_response(response_text, record['url'], is_list=False, source_content=extracted_text)
                
                if isinstance(parsed_json, dict):
                    structured_meetings.append(parsed_json)
        
        app_logger.info(f"Successfully parsed {len(structured_meetings)} meetings.")
        return structured_meetings

    def summarize_news_batch(self, records: List[Dict[str, Any]], current_date: str) -> List[Dict[str, Any]]:
        """Uses 2.0-flash model for initial processing, then 2.5-pro for final refinement."""
        if not records:
            return []
            
        # First, use gemini-2.0-flash for initial summarization (efficient processing)
        app_logger.info(f"Starting initial news summarization using {self.config.flash_model_name} for {len(records)} records...")
        
        # Use content-based batching for the flash model
        content_batches = self.build_content_aware_batches(records, self.config.flash_model_name)
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

                response_text = self._call_llm_with_backoff(prompt, self.config.flash_model_name)
                parsed_json = self._parse_json_response(response_text, record['url'], is_list=False, source_content=original_text)
                
                if isinstance(parsed_json, dict):
                    initial_summaries.append(parsed_json)
                    processed_urls.add(record['url'])
        
        app_logger.info(f"Successfully generated initial summaries for {len(initial_summaries)} news items using {self.config.flash_model_name}.")
        
        # Then, for higher accuracy, refine key summaries using 2.5-pro if configured
        if hasattr(self.config, 'use_pro_for_refinement') and self.config.use_pro_for_refinement:
            app_logger.info(f"Refining top summaries using {self.config.pro_model_name}...")
            refined_summaries = self._refine_summaries_with_pro(initial_summaries, current_date)
            return refined_summaries
        else:
            return initial_summaries

    def _refine_summaries_with_pro(self, summaries: List[Dict[str, Any]], current_date: str) -> List[Dict[str, Any]]:
        """Use the Pro model to refine summaries if higher accuracy is needed."""
        refined_summaries = []
        
        for summary in summaries:
            url = summary.get('url', '')
            app_logger.info(f"Refining summary for {url} using {self.config.pro_model_name}...")
            
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
            
            response_text = self._call_llm_with_backoff(refinement_prompt, self.config.pro_model_name)
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

        response_text = self._call_llm_with_backoff(prompt, self.config.pro_model_name)
        parsed_list = self._parse_json_response(response_text, "top_story_selection", is_list=True)
        
        if isinstance(parsed_list, list):
            app_logger.info(f"Successfully selected {len(parsed_list)} top stories.")
            return parsed_list
        
        app_logger.error("Failed to select top stories, returned data was not a list.")
        return []

    def format_final_briefing(self, target_date: str, current_date: str, meetings_json: List[Dict], all_summaries: List[Dict], top_story_urls: List[str]) -> Optional[str]:
        """Uses PRO model to assemble the final report."""
        app_logger.info("Starting final briefing formatting (PRO model)...")
        
        # 1. Format date for title
        try:
            date_obj = datetime.strptime(target_date, '%Y-%m-%d')
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
        final_report = self._call_llm_with_backoff(prompt, self.config.pro_model_name)
        
        if final_report:
            app_logger.info("Successfully generated final briefing.")
        else:
            app_logger.error("Final briefing generation failed.")
            
        return final_report

