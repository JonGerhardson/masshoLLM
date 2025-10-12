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
    
    prompt = f"""
    You are an automated text processing service. Today's date is {today_date_str}.
    You will be given a series of documents, each identified by a URL and a type ('yes' for new, 'maybe' for unknown date).
    Your task is to process ALL documents and return a single, valid JSON array as your response.
    Do not include any text, pleasantries, or markdown formatting before or after the JSON array.

    For each document:
    1. If the type is 'yes', classify it as "New Document". The justification should be "Confirmed new via site metadata."
    2. If the type is 'maybe', classify the content into ONE of these categories: "New Announcement", "Recent Update", or "Timeless Info", and provide a one-sentence justification.
    3. For ALL documents, provide a concise, neutral, two-sentence summary suitable for a news lead.

    The format for each object in the JSON array MUST be:
    {{
      "url": "The original URL of the document",
      "category": "Your chosen category",
      "justification": "Your one-sentence justification",
      "summary": "Your two-sentence summary"
    }}
    """
    return prompt

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

def _call_gemini_pseudo_batch(batch: List[Dict[str, Any]], api_key: str, model_name: str) -> Dict[str, Dict[str, str]]:
    """Handles a single 'pseudo-batch' API call to Google Gemini for a subset of documents."""
    llm_logger = logging.getLogger('llm')
    original_urls = [item['url'] for item in batch]

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        master_prompt = get_pseudo_batch_prompt()
        
        content_block = []
        for item in batch:
            doc_type = 'yes' if not item['is_maybe'] else 'maybe'
            content_header = f"--- DOCUMENT URL: {item['url']} TYPE: {doc_type} ---"
            item_full_text = f"{content_header}\n{item['content']}"
            content_block.append(item_full_text)

        full_content = "\n\n".join(content_block)
        full_prompt = f"{master_prompt}\n\n--- START OF DOCUMENTS ---\n\n{full_content}"
        
        llm_logger.debug(f"--- GEMINI PSEUDO-BATCH REQUEST ({len(batch)} docs) ---")
        
        generation_config = { "max_output_tokens": 8192 }
        
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        response_text = response.text.strip()
        
        llm_logger.debug(f"--- GEMINI PSEUDO-BATCH RESPONSE ---\n{response_text}\n--------------------")

        return _parse_json_response(response_text, original_urls)

    except Exception as e:
        logging.error(f"Fatal error during Gemini pseudo-batch API call: {e}")
        llm_logger.error(f"Fatal error calling Gemini API: {e}", exc_info=True)
        return {url: {"category": "API Error", "summary": str(e)} for url in original_urls}

def get_batch_summaries(batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """
    Generates summaries for a batch of content using the configured LLM provider.
    Splits the main batch into smaller chunks to avoid API response size limits.
    """
    try:
        config = _load_config()['llm_settings']
    except (TypeError, KeyError):
        logging.error("LLM settings are missing or malformed in config.yaml.")
        return {}

    provider = config.get('provider')
    models = config.get('models', {})
    
    if not batch:
        return {}

    logging.info(f"Requesting {len(batch)} summaries using provider: {provider} (Chunked Pseudo-Batch Mode)")

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or config.get('api_keys', {}).get('gemini')
        model_name = models.get('gemini', 'gemini-2.5-flash')
        
        if not api_key or "YOUR_" in api_key:
            logging.error("Gemini API key is not configured.")
            return {item['url']: {"category": "Config Error", "summary": "API key not configured."} for item in batch}

        all_results = {}
        max_requests = 10
        chunk_size = max(1, math.ceil(len(batch) / max_requests))
        total_chunks = math.ceil(len(batch) / chunk_size)

        logging.info(f"Total batch of {len(batch)} will be split into {total_chunks} chunks of up to {chunk_size} documents each.")

        for i in range(0, len(batch), chunk_size):
            sub_batch = batch[i:i + chunk_size]
            chunk_num = (i // chunk_size) + 1
            
            logging.info(f"Processing chunk {chunk_num}/{total_chunks} with {len(sub_batch)} documents.")
            
            results = _call_gemini_pseudo_batch(sub_batch, api_key, model_name)
            all_results.update(results)
            
            if chunk_num < total_chunks:
                logging.info("Waiting for 2 seconds before next API call...")
                time.sleep(2)

        return all_results

    else:
        logging.error(f"Provider '{provider}' is not configured for pseudo-batching. Aborting.")
        return {item['url']: {"category": "Config Error", "summary": "Provider not configured for pseudo-batching."} for item in batch}

