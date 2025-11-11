import yaml
import logging
import requests
import google.generativeai as genai
import os
import re
import json
import math
import time
from datetime import datetime
from typing import List, Dict, Any

import database

# Import fact checker
try:
    from fact_checker import verify_summary
except ImportError:
    # Fallback if fact_checker is not available
    def verify_summary(source_text: str, summary: str, url: str = "") -> Dict[str, Any]:
        # Dummy function if fact_checker is not available
        return {
            'accuracy_score': 1.0,
            'is_accurate': True,
            'confidence_score': 1.0,
            'issues': [],
            'suggestions': ['Fact checking not available']
        }

# --- Global UserAgent Instance and Logging setup are unchanged ---
from fake_useragent import UserAgent
ua = UserAgent()

def setup_llm_logging(date_str: str):
    log_filename = f"logs/{date_str}_llm.log"
    llm_logger = logging.getLogger('llm')
    llm_logger.setLevel(logging.DEBUG)
    if llm_logger.hasHandlers():
        llm_logger.handlers.clear()
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    llm_logger.addHandler(fh)
    logging.info(f"LLM logging configured to file: {log_filename}")

def get_pseudo_batch_prompt():
    """Generates the system prompt for the LLM to handle a batch and return JSON."""
    today_date_str = datetime.now().strftime("%Y-%m-%d")
    
    # --- PROMPT UPDATED with new categories and logic ---
    prompt = f"""
    You are an automated text processing service. Today's date is {today_date_str}.
    You will be given a series of documents, each identified by a URL and a type ('yes' for new, 'maybe' for unknown date).
    Your task is to process ALL documents and return a single, valid JSON array as your response.
    Do not include any text, pleasantries, or markdown formatting before or after the JSON array.

    CRITICAL: Before categorizing, carefully check if this content references information more than TWO MONTHS old. 
    If the content is clearly outdated (older than 2 months), classify it as "Outdated Content".

    For each document:
    1. If the type is 'yes', classify it as "New Document". The justification should be "Confirmed new via site metadata."
       (Note: Some 'yes' documents might be meeting notices; if the content *clearly* indicates a meeting,
       you may override this and classify as "Meeting Announcement" or "Meeting Materials".)
       However, if it is clearly more than 2 months old, classify as "Outdated Content".
    
    2. If the type is 'maybe', classify the content into ONE of these categories:
       - "Outdated Content": Content that references information more than TWO MONTHS old
       - "Meeting Announcement": The text is a notice for a public meeting scheduled for today, yesterday, or a future date.
       - "Meeting Materials": The text provides materials for a meeting that has already passed (e.g., minutes, agendas, recordings, presentations).
       - "New Announcement": A news-like update, press release, or time-sensitive public notice (NOT a meeting).
       - "Recent Update": A minor change to an existing informational page.
       - "Timeless Info": A general informational page with no clear publication date (e.g., a "how-to" guide, main topic page).

    3. For ALL documents, provide a concise, neutral, two-sentence summary suitable for a news lead.

    The format for each object in the JSON array MUST be:
    {{
      "url": "The original URL of the document",
      "category": "Your chosen category (Outdated Content, New Document, Meeting Announcement, Meeting Materials, New Announcement, Recent Update, or Timeless Info)",
      "justification": "Your one-sentence justification",
      "summary": "Your two-sentence summary"
    }}
    """
    return prompt

def get_summary_only_prompt():
    """Generates a system prompt for the LLM to only summarize content, not categorize it."""
    return (
        "You are an automated text processing service for a news organization. "
        "You will be given a series of documents, each identified by a unique URL. "
        "Your task is to process ALL documents provided and return a single, valid JSON array as your response. "
        "Do not include any text, pleasantries, or markdown formatting before or after the JSON array.\n\n"
        "For each document:\n"
        "1. Write a concise, neutral, two-sentence summary suitable for a news lead.\n"
        "2. Base your summary *only* on the text provided for that document.\n\n"
        "The format for each object in the JSON array MUST be:\n"
        '{\n'
        '  "url": "The original URL of the document",\n'
        '  "summary": "Your two-sentence summary."\n'
        '}'
    )


_config = None
def _load_config():
    """Loads and caches the config.yaml file."""
    global _config
    if _config is None:
        try:
            with open("config.yaml", 'r') as f:
                _config = yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            logging.critical(f"CRITICAL ERROR with config.yaml: {e}")
            raise
    return _config

def _parse_json_response(response_text: str, original_urls: List[str]) -> Dict[str, Dict[str, str]]:
    """Extracts and parses the JSON array from the LLM's response text."""
    llm_logger = logging.getLogger('llm')
    
    match = re.search(r'```json\s*(\[.*\])\s*```|(\[.*\])', response_text, re.DOTALL)
    if not match:
        llm_logger.error(f"Could not find a JSON array in the response.")
        return {url: {"category": "API Error", "summary": "No JSON array found in response."} for url in original_urls}

    json_str = next((g for g in match.groups() if g is not None), None)

    try:
        parsed_data = json.loads(json_str)
        if not isinstance(parsed_data, list):
            raise json.JSONDecodeError("JSON is not a list.", json_str, 0)
        
        results = {}
        for item in parsed_data:
            url = item.get("url")
            if url:
                justification = item.get("justification", "")
                summary_text = item.get("summary", "No summary provided.")
                full_summary = f"{justification}\n\n{summary_text}" if justification and "metadata" not in justification else summary_text
                results[url] = {
                    "category": item.get("category", "Unknown"),
                    "summary": full_summary.strip()
                }
        
        for url in original_urls:
            if url not in results:
                results[url] = {"category": "API Error", "summary": "LLM did not return an entry for this URL in its response."}

        return results
    except json.JSONDecodeError as e:
        llm_logger.error(f"Failed to decode JSON from LLM response: {e}")
        llm_logger.debug(f"Malformed JSON string: {json_str}")
        return {url: {"category": "API Error", "summary": f"Malformed JSON response from LLM: {e}"} for url in original_urls}

def _parse_json_response_with_fact_checking(response_text: str, original_urls: List[str], batch_contents: List[str]) -> Dict[str, Dict[str, str]]:
    """Extracts and parses the JSON array from the LLM's response text with fact checking."""
    llm_logger = logging.getLogger('llm')
    
    match = re.search(r'```json\s*(\[.*\])\s*```|(\[.*\])', response_text, re.DOTALL)
    if not match:
        llm_logger.error(f"Could not find a JSON array in the response.")
        return {url: {"category": "API Error", "summary": "No JSON array found in response."} for url in original_urls}

    json_str = next((g for g in match.groups() if g is not None), None)

    try:
        parsed_data = json.loads(json_str)
        if not isinstance(parsed_data, list):
            raise json.JSONDecodeError("JSON is not a list.", json_str, 0)
        
        results = {}
        url_to_content_map = dict(zip(original_urls, batch_contents))
        
        for item in parsed_data:
            url = item.get("url")
            if url:
                justification = item.get("justification", "")
                summary_text = item.get("summary", "No summary provided.")
                full_summary = f"{justification}\n\n{summary_text}" if justification and "metadata" not in justification else summary_text
                
                # Perform fact checking if we have the source content
                if url in url_to_content_map:
                    source_content = url_to_content_map[url]
                    verification_result = verify_summary(source_content, full_summary, url)
                    
                    if not verification_result.get('is_accurate', True):
                        # Add detailed warning to summary if fact check fails
                        detailed_issues = verification_result.get('detailed_issues', [])
                        if detailed_issues:
                            issues_text = "; ".join(detailed_issues[:2])  # Limit to first 2 issues to avoid clutter
                            if len(detailed_issues) > 2:
                                issues_text += f"; +{len(detailed_issues) - 2} more issues"
                            accuracy_warning = f"\n\n[FACT CHECK WARNING: {len(detailed_issues)} accuracy issues - {issues_text}. {verification_result.get('suggestions', ['Review for accuracy'])[0]}]"
                        else:
                            accuracy_warning = f"\n\n[FACT CHECK WARNING: {len(verification_result.get('issues', []))} accuracy issues detected. {verification_result.get('suggestions', ['Review for accuracy'])[0]}]"
                        full_summary += accuracy_warning
                        llm_logger.warning(f"Fact check failed for {url}: {detailed_issues if detailed_issues else verification_result.get('issues', [])}")
                
                results[url] = {
                    "category": item.get("category", "Unknown"),
                    "summary": full_summary.strip()
                }
        
        for url in original_urls:
            if url not in results:
                results[url] = {"category": "API Error", "summary": "LLM did not return an entry for this URL in its response."}

        return results
    except json.JSONDecodeError as e:
        llm_logger.error(f"Failed to decode JSON from LLM response: {e}")
        llm_logger.debug(f"Malformed JSON string: {json_str}")
        return {url: {"category": "API Error", "summary": f"Malformed JSON response from LLM: {e}"} for url in original_urls}

def _parse_summary_only_json_response(response_text: str, original_urls: List[str]) -> Dict[str, str]:
    """Extracts and parses a summary-only JSON array from the LLM's response."""
    llm_logger = logging.getLogger('llm')
    
    match = re.search(r'```json\s*(\[.*\])\s*```|(\[.*\])', response_text, re.DOTALL)
    if not match:
        llm_logger.error("Could not find a JSON array in the summary-only response.")
        return {url: "API Error: No JSON array found." for url in original_urls}

    json_str = next((g for g in match.groups() if g is not None), None)

    try:
        parsed_data = json.loads(json_str)
        if not isinstance(parsed_data, list):
            raise json.JSONDecodeError("JSON is not a list.", json_str, 0)
        
        results = {item.get("url"): item.get("summary", "No summary provided.") for item in parsed_data if item.get("url")}
        
        for url in original_urls:
            if url not in results:
                results[url] = "API Error: LLM did not return an entry for this URL."
        return results
    except json.JSONDecodeError as e:
        llm_logger.error(f"Failed to decode summary-only JSON: {e}")
        return {url: "API Error: Malformed JSON response." for url in original_urls}

def _call_gemini_pseudo_batch(batch: List[Dict[str, Any]], api_key: str, model_name: str, master_prompt: str, parser_func: callable) -> Dict[str, Any]:
    """Handles a single generic 'pseudo-batch' API call to Google Gemini."""
    llm_logger = logging.getLogger('llm')
    original_urls = [item['url'] for item in batch]

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # Calculate content size and apply appropriate batching
        total_content_size = sum(len(item.get('content', '')) for item in batch)
        llm_logger.debug(f"Processing batch with {len(batch)} items, total size: {total_content_size} chars")
        
        content_block = []
        batch_contents = []  # Keep track of original content for fact checking
        for item in batch:
            # For the original prompt, include the type. For others, just the URL.
            doc_type_str = f" TYPE: {'yes' if not item.get('is_maybe') else 'maybe'}" if 'is_maybe' in item else ""
            content_header = f"--- DOCUMENT URL: {item['url']}{doc_type_str} ---"
            item_full_text = f"{content_header}\n{item['content']}"
            content_block.append(item_full_text)
            batch_contents.append(item['content'])  # Store original content

        full_content = "\n\n".join(content_block)
        full_prompt = f"{master_prompt}\n\n--- START OF DOCUMENTS ---\n\n{full_content}"
        
        llm_logger.debug(f"--- GEMINI PSEUDO-BATCH REQUEST ({len(batch)} docs) ---")
        
        generation_config = {"max_output_tokens": 8192}
        
        response = model.generate_content(full_prompt, generation_config=generation_config)
        response_text = response.text.strip()
        
        llm_logger.debug(f"--- GEMINI PSEUDO-BATCH RESPONSE ---\n{response_text}\n--------------------")

        # Use fact-checking parser if available and appropriate
        if parser_func == _parse_json_response:
            return _parse_json_response_with_fact_checking(response_text, original_urls, batch_contents)
        else:
            return parser_func(response_text, original_urls)

    except Exception as e:
        logging.error(f"Fatal error during Gemini pseudo-batch API call: {e}")
        llm_logger.error(f"Fatal error calling Gemini API: {e}", exc_info=True)
        # Adjust error format based on the expected parser output
        if parser_func == _parse_summary_only_json_response:
             return {url: f"API Error: {e}" for url in original_urls}
        else:
             return {url: {"category": "API Error", "summary": str(e)} for url in original_urls}


def calculate_optimal_batch_size_for_model(model_name: str) -> tuple:
    """
    Calculate optimal batch size based on model-specific rate limits and TPM
    """
    # Model-specific limits
    model_limits = {
        'gemini-2.0-flash': {'rpm': 15, 'tpm': 1000000},  # 1M tokens per minute
        'gemini-2.5-pro': {'rpm': 2, 'tpm': 125000},      # 125K tokens per minute
        'gemini-2.5-flash': {'rpm': 10, 'tpm': 250000},    # 250K tokens per minute
        'gemini-1.5-pro': {'rpm': 2, 'tpm': 125000},      # 125K tokens per minute
        'gemini-1.5-flash': {'rpm': 15, 'tpm': 1000000}   # 1M tokens per minute
    }
    
    limits = model_limits.get(model_name, model_limits['gemini-2.0-flash'])
    
    if limits['rpm'] <= 2:  # For low RPM models like 2.5-pro
        max_chars_per_batch = min(70000, limits['tpm'] // 3)  # Use 1/3 of TPM budget per batch
        max_items_per_batch = 2  # Very small batch count due to low RPM
    elif limits['rpm'] <= 10:  # For medium RPM models like 2.5-flash
        max_chars_per_batch = min(150000, limits['tpm'] // 4)  # Use 1/4 of TPM budget per batch
        max_items_per_batch = 4
    else:  # For high RPM models like 2.0-flash
        max_chars_per_batch = min(600000, limits['tpm'] // 8)  # Use 1/8 of TPM budget per batch
        max_items_per_batch = 8
    
    return max_chars_per_batch, max_items_per_batch

def build_content_aware_batches(items: List[Dict[str, Any]], max_chars_per_batch: int, max_items_per_batch: int) -> List[List[Dict[str, Any]]]:
    """
    Build batches based on content size rather than fixed count
    """
    if not items:
        return []
    
    # Sort items by content length (descending) to better pack batches
    sorted_items = sorted(items, key=lambda x: len(x.get('content', '')), reverse=True)
    
    batches = []
    current_batch = []
    current_batch_size = 0
    
    for item in sorted_items:
        item_content = item.get('content', '') or ''
        item_size = len(item_content)
        
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
                # Put very large content in its own batch
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

def get_batch_summaries(batch: List[Dict[str, Any]], db_connection=None, table_name: str = None) -> Dict[str, Dict[str, str]]:
    """
    Generates summaries AND categories for sitemap items using content-based batching.
    """
    try:
        config = _load_config()['llm_settings']
    except (TypeError, KeyError):
        logging.error("LLM settings are missing or malformed in config.yaml.")
        return {}
        
    api_key = os.environ.get("GEMINI_API_KEY") or config.get('api_keys', {}).get('gemini')
    model_name = config.get('models', {}).get('gemini', 'gemini-2.0-flash')  # Use 2.0-flash as default for efficiency
    
    if not api_key or "YOUR_" in api_key:
        logging.error("Gemini API key is not configured.")
        return {item['url']: {"category": "Config Error", "summary": "API key not configured."} for item in batch}

    # Calculate optimal batching based on the model
    max_chars_per_batch, max_items_per_batch = calculate_optimal_batch_size_for_model(model_name)
    logging.info(f"Using model {model_name} with max {max_chars_per_batch} chars and {max_items_per_batch} items per batch")
    
    # Build content-aware batches
    content_batches = build_content_aware_batches(batch, max_chars_per_batch, max_items_per_batch)
    all_results = {}

    total_batches = len(content_batches)
    logging.info(f"Processing {len(batch)} items in {total_batches} content-aware batches (optimized for {model_name})")
    
    for i, sub_batch in enumerate(content_batches):
        batch_num = i + 1
        
        # Calculate batch statistics
        batch_char_count = sum(len(item.get('content', '')) for item in sub_batch)
        logging.info(f"Processing batch {batch_num}/{total_batches} "
                    f"({len(sub_batch)} items, ~{batch_char_count} chars)...")
        
        results = _call_gemini_pseudo_batch(sub_batch, api_key, model_name, 
                                          get_pseudo_batch_prompt(), _parse_json_response)
        all_results.update(results)
        
        # Adjust delay based on model's RPM (higher for low-RPM models)
        if 'gemini-2.5-pro' in model_name or 'gemini-1.5-pro' in model_name:
            time.sleep(35)  # More than 30 seconds to respect 2 RPM limit
        elif 'gemini-2.5-flash' in model_name:
            time.sleep(7)   # About 6 seconds to respect 10 RPM limit
        else:  # gemini-2.0-flash and others with higher RPM
            time.sleep(5)   # About 4 seconds to respect 15 RPM limit

    # Mark outdated content in DB if database connection is provided
    if db_connection and table_name:
        mark_outdated_content_in_db_from_results(db_connection, table_name, all_results)
    
    return all_results

def mark_outdated_content_in_db_from_results(db_connection, table_name: str, results: Dict[str, Dict[str, str]]):
    """Mark URLs with 'Outdated Content' category as excluded in the database."""
    if not db_connection or not table_name:
        return

    cursor = db_connection.cursor()
    
    outdated_urls = []
    for url, result in results.items():
        if result.get('category') == 'Outdated Content':
            outdated_urls.append(url)
    
    for url in outdated_urls:
        try:
            # Update the record to mark it as excluded
            update_sql = f"UPDATE {table_name} SET excluded = 'yes' WHERE url = ?"
            cursor.execute(update_sql, (url,))
            logging.info(f"Marked {url} as excluded due to outdated content")
        except Exception as e:
            logging.error(f"Failed to mark {url} as excluded: {e}")
    
    db_connection.commit()
    logging.info(f"Updated database to mark {len(outdated_urls)} records as excluded")

def get_press_release_summaries(batch: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Generates SUMMARIES ONLY for press release items using content-based batching.
    """
    try:
        config = _load_config()['llm_settings']
    except (TypeError, KeyError):
        logging.error("LLM settings are missing or malformed in config.yaml.")
        return {}

    api_key = os.environ.get("GEMINI_API_KEY") or config.get('api_keys', {}).get('gemini')
    model_name = config.get('models', {}).get('gemini', 'gemini-2.0-flash')  # Use 2.0-flash for efficiency
    
    if not api_key or "YOUR_" in api_key:
        logging.error("Gemini API key is not configured.")
        return {item['url']: "API key not configured." for item in batch}

    # Calculate optimal batching for this model
    max_chars_per_batch, max_items_per_batch = calculate_optimal_batch_size_for_model(model_name)
    logging.info(f"Using model {model_name} with max {max_chars_per_batch} chars and {max_items_per_batch} items per batch for press releases")
    
    # Build content-aware batches for press releases
    content_batches = build_content_aware_batches(batch, max_chars_per_batch, max_items_per_batch)
    all_results = {}

    total_batches = len(content_batches)
    logging.info(f"Processing {len(batch)} press releases in {total_batches} content-aware batches")
    
    for i, sub_batch in enumerate(content_batches):
        batch_num = i + 1
        
        # Calculate batch statistics
        batch_char_count = sum(len(item.get('content', '')) for item in sub_batch)
        logging.info(f"Processing press release batch {batch_num}/{total_batches} "
                    f"({len(sub_batch)} items, ~{batch_char_count} chars)...")
        
        results = _call_gemini_pseudo_batch(sub_batch, api_key, model_name, 
                                          get_summary_only_prompt(), _parse_summary_only_json_response)
        all_results.update(results)
        
        # Adjust delay based on model's RPM
        if 'gemini-2.5-pro' in model_name or 'gemini-1.5-pro' in model_name:
            time.sleep(35)  # More than 30 seconds to respect 2 RPM limit
        elif 'gemini-2.5-flash' in model_name:
            time.sleep(7)   # About 6 seconds to respect 10 RPM limit
        else:  # gemini-2.0-flash and others with higher RPM
            time.sleep(5)   # About 4 seconds to respect 15 RPM limit
    
    return all_results

