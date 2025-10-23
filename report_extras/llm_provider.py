import logging
import yaml
import os
import re
import json
import time
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from typing import Optional, List, Dict, Any, Union

# --- Logging Setup ---
app_logger = logging.getLogger('briefing_app')
llm_logger = logging.getLogger('briefing_llm')

class BriefingConfig:
    """Holds the configuration for the briefing generation script."""
    def __init__(self, config_dict: Dict):
        config_dict = config_dict or {}
        llm_settings = config_dict.get('llm_settings', {})
        
        # Model names
        self.pro_model_name = llm_settings.get('pro_model', 'gemini-2.5-pro')
        self.flash_model_name = llm_settings.get('flash_model', 'gemini-2.5-flash')
        
        # Rate Limits (RPM)
        self.pro_rpm = llm_settings.get('pro_rpm', 2)
        self.flash_rpm = llm_settings.get('flash_rpm', 10)
        
        # Calculate delay in seconds based on RPM
        self.pro_delay_sec = 60.0 / self.pro_rpm
        self.flash_delay_sec = 60.0 / self.flash_rpm
        
        # Text processing
        self.truncation_length = llm_settings.get('truncation_length', 15000)
        
        # Retry logic
        self.max_retries = llm_settings.get('max_retries', 3)
        self.initial_backoff_sec = llm_settings.get('initial_backoff_sec', 60)

class LLMProvider:
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

    def _apply_rate_limit(self, is_pro_model: bool):
        """Enforces a simple time-based rate limit."""
        if is_pro_model:
            time_since_last_call = time.time() - self.last_pro_call_time
            wait_time = self.config.pro_delay_sec - time_since_last_call
            if wait_time > 0:
                app_logger.info(f"PRO rate limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            self.last_pro_call_time = time.time()
        else:
            time_since_last_call = time.time() - self.last_flash_call_time
            wait_time = self.config.flash_delay_sec - time_since_last_call
            if wait_time > 0:
                app_logger.debug(f"FLASH rate limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            self.last_flash_call_time = time.time()

    def _call_gemini_with_backoff(self, model: genai.GenerativeModel, prompt: str, is_pro_model: bool) -> Optional[str]:
        """Makes an API call with rate limiting and exponential backoff."""
        self._apply_rate_limit(is_pro_model)
        
        for attempt in range(self.config.max_retries):
            try:
                llm_logger.debug(f"--- LLM REQUEST (Attempt {attempt + 1}) ---\n{prompt[:1000]}...")
                response = model.generate_content(prompt)
                response_text = response.text.strip()
                llm_logger.debug(f"--- LLM RESPONSE ---\n{response_text}")
                return response_text
            
            except google_exceptions.ResourceExhausted as e:
                if attempt < self.config.max_retries - 1:
                    delay = self.config.initial_backoff_sec * (2 ** attempt)
                    app_logger.warning(f"Quota exceeded. Retrying in {delay} seconds... (Attempt {attempt + 1}/{self.config.max_retries})")
                    time.sleep(delay)
                else:
                    app_logger.error(f"Fatal error: Quota exceeded after {self.config.max_retries} retries.", exc_info=True)
                    return None
            
            except Exception as e:
                app_logger.error(f"Fatal error during Gemini API call: {e}", exc_info=True)
                return None
        
        return None

    def _parse_json_response(self, text: str, url: str, is_list: bool = False) -> Optional[Union[Dict, List]]:
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
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            app_logger.error(f"Failed to decode JSON from LLM response for {url}: {e}")
            llm_logger.error(f"MALFORMED JSON:\n{json_str}")
            return None

    def parse_meeting_batch(self, records: List[Dict[str, Any]], current_date: str) -> List[Dict[str, Any]]:
        """Uses FLASH model to parse meeting announcements into structured JSON."""
        app_logger.info(f"Starting meeting parsing batch for {len(records)} records...")
        structured_meetings = []
        
        for i, record in enumerate(records):
            app_logger.info(f"Parsing meeting {i+1}/{len(records)}: {record['url']}")
            
            prompt_template = self.prompts['1_parse_meeting']
            prompt = prompt_template.replace("[CURRENT_DATE]", current_date)
            prompt = prompt.replace("[MEETING_URL]", record['url'])
            prompt = prompt.replace("[MEETING_TEXT]", record.get('extracted_text', ''))
            
            response_text = self._call_gemini_with_backoff(self.flash_model, prompt, is_pro_model=False)
            parsed_json = self._parse_json_response(response_text, record['url'], is_list=False)
            
            if isinstance(parsed_json, dict):
                structured_meetings.append(parsed_json)
        
        app_logger.info(f"Successfully parsed {len(structured_meetings)} meetings.")
        return structured_meetings

    def summarize_news_batch(self, records: List[Dict[str, Any]], current_date: str) -> List[Dict[str, Any]]:
        """Uses FLASH model to summarize news items into structured JSON."""
        app_logger.info(f"Starting news summary batch for {len(records)} records...")
        summaries = []

        for i, record in enumerate(records):
            app_logger.info(f"Summarizing news {i+1}/{len(records)}: {record['url']}")
            
            text = record.get('extracted_text', '')
            
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

            response_text = self._call_gemini_with_backoff(self.flash_model, prompt, is_pro_model=False)
            parsed_json = self._parse_json_response(response_text, record['url'], is_list=False)
            
            if isinstance(parsed_json, dict):
                summaries.append(parsed_json)
        
        app_logger.info(f"Successfully summarized {len(summaries)} news items.")
        return summaries

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

        response_text = self._call_gemini_with_backoff(self.pro_model, prompt, is_pro_model=True)
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
        final_report = self._call_gemini_with_backoff(self.pro_model, prompt, is_pro_model=True)
        
        if final_report:
            app_logger.info("Successfully generated final briefing.")
        else:
            app_logger.error("Final briefing generation failed.")
            
        return final_report
