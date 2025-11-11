import logging
import time
import json
import re
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

# --- Logging Setup ---
app_logger = logging.getLogger('briefing_app')
llm_logger = logging.getLogger('briefing_llm')

class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    Defines the interface that all LLM providers must implement.
    """
    
    @abstractmethod
    def __init__(self, config, prompts: Dict[str, str]):
        """
        Initialize the provider with configuration and prompts.
        """
        pass

    @abstractmethod
    def parse_meeting_batch(self, records: List[Dict[str, Any]], current_date: str) -> List[Dict[str, Any]]:
        """
        Parse meeting announcements into structured JSON.
        """
        pass

    @abstractmethod
    def summarize_news_batch(self, records: List[Dict[str, Any]], current_date: str) -> List[Dict[str, Any]]:
        """
        Summarize news items.
        """
        pass

    @abstractmethod
    def select_top_stories(self, all_summaries: List[Dict[str, Any]], current_date: str) -> List[str]:
        """
        Select top stories.
        """
        pass

    @abstractmethod
    def format_final_briefing(self, target_date: str, current_date: str, meetings_json: List[Dict], 
                             all_summaries: List[Dict], top_story_urls: List[str]) -> Optional[str]:
        """
        Format the final briefing.
        """
        pass

    @abstractmethod
    def _call_llm_with_backoff(self, prompt: str, model_name: str) -> Optional[str]:
        """
        Make an LLM call with rate limiting and exponential backoff.
        """
        pass

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
            return parsed_result
        except json.JSONDecodeError as e:
            app_logger.error(f"Failed to decode JSON from LLM response for {url}: {e}")
            llm_logger.error(f"MALFORMED JSON:\n{json_str}")
            return None

    def calculate_optimal_batch_size(self, model_name: str) -> tuple:
        """
        Calculate optimal batch size based on model-specific rate limits and TPM
        (This is a default implementation, can be overridden by specific providers)
        """
        # Default values, may need to be customized per provider
        max_chars_per_batch = 80000
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