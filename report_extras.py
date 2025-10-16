import logging
import argparse
import yaml
import os
import re
import time
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import math

# --- Import the database module to interact with the database ---
import database

# --- Default Constants (can be overridden by config.yaml) ---
API_CALL_DELAY_SECONDS = 20 # Increased delay to better respect API rate limits.
MODEL_NAME = 'gemini-2.5-pro' # Reverted to original model per user request.
FINAL_FORMAT_TEMPERATURE = 0.2
URL_CHUNK_SIZE = 5 # Number of pages to combine into a single API request
MAX_TEXT_LENGTH_CHARS = 500000 # Sanity check for extremely long text.
TRUNCATION_LENGTH_CHARS = 10000 # Limit for excerpting long text passed to the LLM.

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BriefingConfig:
    """Holds the configuration for the briefing generation script."""
    def __init__(self, config_dict: Dict):
        config_dict = config_dict or {}
        llm_settings = config_dict.get('llm_settings', {})
        self.api_call_delay = llm_settings.get('api_call_delay', API_CALL_DELAY_SECONDS)
        self.model_name = llm_settings.get('model_name', MODEL_NAME)
        self.final_format_temperature = llm_settings.get('final_format_temperature', FINAL_FORMAT_TEMPERATURE)
        self.url_chunk_size = llm_settings.get('url_chunk_size', URL_CHUNK_SIZE)
        self.max_text_length = llm_settings.get('max_text_length', MAX_TEXT_LENGTH_CHARS)
        self.truncation_length = llm_settings.get('truncation_length', TRUNCATION_LENGTH_CHARS)

def load_config() -> Optional[BriefingConfig]:
    """Loads and validates the configuration from config.yaml."""
    try:
        with open("config.yaml", 'r') as f:
            config_dict = yaml.safe_load(f)
            return BriefingConfig(config_dict)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logging.critical(f"CRITICAL ERROR reading config.yaml: {e}")
        return None

def get_pseudo_batch_summary_prompt() -> str:
    """Generates the system prompt for the LLM to handle a batch of summaries and return JSON."""
    return (
        "You are an automated text processing service for a news organization. "
        "You will be given a series of documents, each identified by a unique URL. "
        "Your task is to process ALL documents provided and return a single, valid JSON array as your response. "
        "Do not include any text, pleasantries, or markdown formatting before or after the JSON array.\n\n"
        "For each document:\n"
        "1. Write a detailed, journalistic paragraph summarizing its key information.\n"
        "2. Base your summary *only* on the text provided for that document.\n"
        "3. Maintain a neutral and factual tone.\n\n"
        "The format for each object in the JSON array MUST be:\n"
        '{\n'
        '  "url": "The original URL of the document",\n'
        '  "summary": "Your detailed summary paragraph."\n'
        '}'
    )

def _parse_batch_json_response(response_text: str, original_urls: List[str]) -> Dict[str, str]:
    """Extracts and parses the JSON array of summaries from the LLM's response text."""
    match = re.search(r'```json\s*(\[.*\])\s*```|(\[.*\])', response_text, re.DOTALL)
    if not match:
        logging.error("Could not find a JSON array in the batch response.")
        return {url: "API Error: No JSON array found in response." for url in original_urls}

    json_str = next((g for g in match.groups() if g is not None), None)

    try:
        parsed_data = json.loads(json_str)
        if not isinstance(parsed_data, list):
            raise json.JSONDecodeError("JSON is not a list.", json_str, 0)
        
        results = {item.get("url"): item.get("summary", "No summary provided.") for item in parsed_data if item.get("url")}
        
        # Ensure all original URLs have a result, even if it's an error.
        for url in original_urls:
            if url not in results:
                results[url] = "API Error: LLM did not return an entry for this URL."

        return results
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from LLM batch response: {e}")
        return {url: f"API Error: Malformed JSON response from LLM: {e}" for url in original_urls}

def _call_gemini_for_batch_summary(batch: List[Dict[str, Any]], model: genai.GenerativeModel, config: BriefingConfig) -> Dict[str, str]:
    """Handles a single 'pseudo-batch' API call to Google Gemini with a retry mechanism."""
    original_urls = [item['url'] for item in batch]
    master_prompt = get_pseudo_batch_summary_prompt()
    
    content_block = []
    for item in batch:
        content_header = f"--- DOCUMENT URL: {item['url']} ---"
        
        # Ensure extracted_text is a string, even if the database field is NULL.
        extracted_text = item.get('extracted_text') or ''
        
        # Truncate extremely long text to avoid excessive token usage
        if len(extracted_text) > config.truncation_length:
            truncation_note = (
                "[NOTE TO EDITOR: The following document was excerpted due to extreme length. "
                "The full text may warrant further review.]\n\n"
            )
            processed_text = truncation_note + extracted_text[:config.truncation_length]
        else:
            processed_text = extracted_text

        item_full_text = f"{content_header}\n{processed_text}"
        content_block.append(item_full_text)

    full_content = "\n\n".join(content_block)
    full_prompt = f"{master_prompt}\n\n--- START OF DOCUMENTS ---\n\n{full_content}"
    
    max_retries = 3
    initial_delay = 60  # Start with a 60-second delay, as suggested by the API error

    for attempt in range(max_retries):
        try:
            logging.debug(f"--- GEMINI BATCH SUMMARY REQUEST ({len(batch)} docs, Attempt {attempt + 1}) ---")
            response = model.generate_content(full_prompt)
            response_text = response.text.strip()
            return _parse_batch_json_response(response_text, original_urls)

        except google_exceptions.ResourceExhausted as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                logging.warning(f"Quota exceeded. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                logging.error(f"Fatal error: Quota exceeded after {max_retries} retries.", exc_info=True)
                return {url: f"API Error: Quota exceeded after multiple retries. {e}" for url in original_urls}
        
        except Exception as e:
            logging.error(f"Fatal error during Gemini batch summary API call: {e}", exc_info=True)
            return {url: f"API Error: {e}" for url in original_urls}

    # This part should not be reached if the loop is structured correctly
    return {url: "API Error: All retry attempts failed." for url in original_urls}

def get_summaries_in_batches(records: List[Dict[str, Any]], model: genai.GenerativeModel, config: BriefingConfig) -> Dict[str, str]:
    """
    Generates summaries for a list of records by processing them in chunks.
    """
    if not records:
        return {}

    all_results = {}
    chunk_size = config.url_chunk_size
    api_call_delay = config.api_call_delay
    total_chunks = math.ceil(len(records) / chunk_size)
    logging.info(f"Total batch of {len(records)} records will be split into {total_chunks} chunks of up to {chunk_size} each.")

    for i in range(0, len(records), chunk_size):
        sub_batch = records[i:i + chunk_size]
        chunk_num = (i // chunk_size) + 1
        
        logging.info(f"Processing chunk {chunk_num}/{total_chunks} with {len(sub_batch)} documents.")
        
        results = _call_gemini_for_batch_summary(sub_batch, model, config)
        all_results.update(results)
        
        if chunk_num < total_chunks:
            logging.info(f"Waiting for {api_call_delay} seconds before next API call...")
            time.sleep(api_call_delay)

    return all_results

def format_final_briefing(combined_parts: str, model: genai.GenerativeModel, temperature: float) -> Optional[str]:
    """Sends the combined parts to Gemini for final formatting."""
    try:
        prompt = (
            "You are a senior news editor. The following text contains a series of raw, detailed notes for a news briefing. "
            "Each note is preceded by a `[SOURCE: <URL>]` marker that indicates its origin.\n\n"
            "Your task is to synthesize, edit, and format these notes into a single, cohesive, and professional news briefing document. "
            "Follow these rules:\n"
            "1. Organize the content logically under clear markdown headings (e.g., `## Top Stories`, `## Health Department Updates`).\n"
            "2. When you reference information from a specific note, you MUST include a markdown hyperlink to the corresponding source URL that was provided directly before that note.\n"
            "3. Do not invent new information. Every key point in your briefing must be traceable back to the raw notes provided.\n"
            "4. Ensure a consistent, journalistic tone throughout.\n"
            "5. If a note begins with `[NOTE TO EDITOR: ...]`, this indicates the source document was too long and has been truncated. In your final formatted summary for that item, you MUST explicitly mention this. For example, add a sentence like: `Editor's Note: The source document for this item was extensive and has been summarized from an excerpt. A full review may be warranted for additional details.` Adapt the wording to fit the context and make a judgment on its importance.\n\n"
            "--- RAW BRIEFING NOTES ---\n"
        )
        
        generation_config = genai.GenerationConfig(temperature=temperature)
        
        logging.info("Sending combined content to Gemini for final formatting...")
        response = model.generate_content(prompt + combined_parts, generation_config=generation_config)
        return response.text
    except Exception as e:
        logging.error(f"Final formatting API call failed: {e}", exc_info=True)
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate a news briefing from a daily report.")
    parser.add_argument("date", help="The date of the report to process, in YYYY-MM-DD format.")
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory for generated files.")
    parser.add_argument("--include-maybe", action="store_true", help="Include pages marked as 'maybe' in the LLM processing.")
    args = parser.parse_args()
    
    if not re.match(r'\d{4}-\d{2}-\d{2}', args.date):
        logging.critical(f"Invalid date format: '{args.date}'. Please use YYYY-MM-DD.")
        return

    app_config = load_config()
    if not app_config:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    briefing_filename = output_dir / f"briefing_{args.date}.md"
    raw_filename = output_dir / f"briefing_{args.date}_full_raw.md"

    logging.info(f"--- Starting News Briefing Generation for {args.date} ---")
        
    api_key_from_env = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key_from_env:
        logging.critical("Neither GEMINI_API_KEY nor GOOGLE_API_KEY environment variable is set.")
        return

    genai.configure(api_key=api_key_from_env)
    model = genai.GenerativeModel(model_name=app_config.model_name)

    # --- Retrieve data directly from the database ---
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
            db_file = config['database_settings']['database_file']
    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        logging.critical(f"Could not load database configuration from config.yaml: {e}")
        return

    conn = database.create_connection(db_file)
    if not conn:
        logging.critical("Failed to create a database connection.")
        return

    table_name = f"massgov_{args.date.replace('-', '_')}"
    
    # --- Conditionally fetch records based on the --include-maybe flag ---
    if args.include_maybe:
        logging.info(f"Fetching 'new' and 'maybe' records from table '{table_name}' for briefing...")
        records_to_process = database.fetch_new_and_maybe_records(conn, table_name)
    else:
        logging.info(f"Fetching only 'new' records from table '{table_name}' for briefing...")
        records_to_process = database.fetch_new_records(conn, table_name)
    
    conn.close()
    
    if not records_to_process:
        log_message = "No new items"
        if args.include_maybe:
            log_message = "No new or potentially new items"
        logging.warning(f"{log_message} were found in the database for {args.date}. Cannot generate briefing.")
        return

    # --- Sanity check for extremely long text entries ---
    for record in records_to_process:
        text_length = len(record.get('extracted_text') or "")
        if text_length > app_config.max_text_length:
            logging.warning(
                f"Document text for {record.get('url')} is extremely long ({text_length} chars) "
                f"and may cause issues or high costs."
            )

    # --- Warn user if the number of API calls is large ---
    total_chunks = math.ceil(len(records_to_process) / app_config.url_chunk_size)
    if total_chunks > 25:
        print(f"WARNING: This operation will result in approximately {total_chunks} API calls.")
        proceed = input("Do you want to proceed? (y/n): ").lower().strip()
        if proceed != 'y':
            logging.info("Operation cancelled by user.")
            return

    # --- Process records in batches ---
    summaries = get_summaries_in_batches(records_to_process, model, app_config)

    briefing_parts = []
    for record in records_to_process:
        url = record.get('url')
        summary_text = summaries.get(url)
        if summary_text and "API Error" not in summary_text:
            briefing_parts.append(f"[SOURCE: {url}]\n{summary_text}")
        else:
            logging.error(f"Failed to get a valid summary for URL {url}: {summary_text or 'Not found in results.'}")
    
    if not briefing_parts:
        logging.error("Failed to generate any valid briefing parts from the API.")
        return

    combined_content = "\n\n---\n\n".join(briefing_parts)
    
    try:
        with open(raw_filename, 'w', encoding='utf-8') as f:
            f.write(combined_content)
        logging.info(f"Successfully saved raw briefing content to {raw_filename}")
    except IOError as e:
        logging.error(f"Failed to write raw briefing file: {e}")

    final_briefing = format_final_briefing(combined_content, model, app_config.final_format_temperature)

    if final_briefing:
        try:
            with open(briefing_filename, 'w', encoding='utf-8') as f:
                f.write(final_briefing)
            logging.info(f"Successfully saved final news briefing to {briefing_filename}")
        except IOError as e:
            logging.error(f"Failed to write briefing file: {e}")
    else:
        logging.error("Failed to generate the final news briefing.")

    logging.info("--- News Briefing Generation Complete ---")

if __name__ == "__main__":
    main()
