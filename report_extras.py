import logging
import argparse
import yaml
import os
import re
import json
import sys
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import math

# --- Resolve Paths ---
# Get the directory of this script (report_extras/)
SCRIPT_DIR = Path(__file__).resolve().parent
# Get the parent directory (the project root)
ROOT_DIR = SCRIPT_DIR.parent
# Add the project root to the Python path to find 'database'
sys.path.append(str(ROOT_DIR))

# --- Custom Modules ---
import database
from llm_provider import BriefingConfig # Provider config
from provider_factory import get_provider # Provider factory

def setup_briefing_logging(date_str: str):
    """Configures logging for the briefing generator, including a detailed LLM log."""
    # Log files will be created in the root 'report_extras_logs'
    log_dir = ROOT_DIR / "report_extras_logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_log_filename = f"{timestamp}_{date_str.replace('-', '')}"
    
    # --- Main Application Logger (briefing_app) ---
    app_log_file = log_dir / f"{base_log_filename}_app.log"
    app_logger = logging.getLogger('briefing_app')
    app_logger.setLevel(logging.INFO)
    if app_logger.hasHandlers():
        app_logger.handlers.clear()
        
    app_fh = logging.FileHandler(app_log_file)
    app_fh.setLevel(logging.INFO)
    app_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    app_fh.setFormatter(app_formatter)
    app_logger.addHandler(app_fh)
    app_logger.addHandler(logging.StreamHandler())

    # --- Detailed LLM Logger (briefing_llm) ---
    llm_log_file = log_dir / f"{base_log_filename}_llm.log"
    llm_logger = logging.getLogger('briefing_llm')
    llm_logger.setLevel(logging.DEBUG)
    if llm_logger.hasHandlers():
        llm_logger.handlers.clear()

    llm_fh = logging.FileHandler(llm_log_file)
    llm_fh.setLevel(logging.DEBUG)
    llm_formatter = logging.Formatter('%(asctime)s - %(message)s')
    llm_fh.setFormatter(llm_formatter)
    llm_logger.addHandler(llm_fh)
    
    app_logger.info(f"Main log file: {app_log_file}")
    app_logger.info(f"LLM log file: {llm_log_file}")
    return app_logger, llm_logger

def load_prompts(prompt_dir: Path = SCRIPT_DIR / "prompts") -> Dict[str, str]:
    """Loads all prompt templates from the prompts directory."""
    prompts = {}
    prompt_files = [
        "1_parse_meeting.txt",
        "2_summarize_news.txt",
        "3_select_top_stories.txt",
        "4_format_final_briefing.txt"
    ]
    try:
        for f_name in prompt_files:
            key = f_name.split('.')[0]
            with open(prompt_dir / f_name, 'r', encoding='utf-8') as f:
                prompts[key] = f.read()
        logging.info(f"Successfully loaded all prompt templates from {prompt_dir}")
        return prompts
    except FileNotFoundError as e:
        logging.critical(f"CRITICAL ERROR: Could not find prompt file: {e.filename}")
        raise
    except Exception as e:
        logging.critical(f"CRITICAL ERROR: Failed to load prompts: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Generate the 'Massachusetts Logfiler' briefing from a daily report.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("date", help="The date of the report to process, in YYYY-MM-DD format.")
    # Output dir default is '.', which will be resolved relative to ROOT_DIR
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory for generated files (relative to project root).")
    parser.add_argument("--include-maybe", action="store_true", help="Include pages marked as 'maybe' in the LLM processing.")
    args = parser.parse_args()
    
    if not re.match(r'\d{4}-\d{2}-\d{2}', args.date):
        print("CRITICAL: Invalid date format. Please use YYYY-MM-DD.")
        return

    # --- Setup ---
    app_logger, llm_logger = setup_briefing_logging(args.date)
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # Load config from project root
        config_path = ROOT_DIR / "config.yaml"
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        app_logger.info(f"Loaded config from {config_path}")
    except (FileNotFoundError, yaml.YAMLError) as e:
        app_logger.critical(f"Could not load config.yaml: {e}")
        return

    try:
        # Prompts are loaded from this script's directory
        prompts = load_prompts()
    except Exception:
        return # Error already logged by load_prompts

    config = BriefingConfig(full_config)
    
    # --- Database Retrieval ---
    # Use hardcoded path to DB in project root, as requested
    db_path = ROOT_DIR / "massgov_updates.db"
    conn = database.create_connection(str(db_path))
    if not conn: 
        app_logger.critical(f"Failed to connect to database at {db_path}")
        return
    
    app_logger.info(f"Connected to database: {db_path}")
    table_name = f"massgov_{args.date.replace('-', '_')}"
    if args.include_maybe:
        app_logger.info(f"Fetching 'new' and 'maybe' records from table '{table_name}'...")
        all_records = database.fetch_new_and_maybe_records(conn, table_name)
    else:
        app_logger.info(f"Fetching only 'new' records from table '{table_name}'...")
        all_records = database.fetch_new_records(conn, table_name)
    conn.close()
    
    if not all_records:
        app_logger.warning(f"No new items found in the database for {args.date}. Cannot generate briefing.")
        return

    # --- NEW LOGIC: Separate Unpublished Records ---
    # Filter records based on the '--unpublished' text
    llm_eligible_records = [r for r in all_records if r.get('extracted_text') != '--unpublished']
    unpublished_records = [r for r in all_records if r.get('extracted_text') == '--unpublished']
    
    app_logger.info(f"Found {len(unpublished_records)} unpublished records to exclude from LLM processing.")

    # --- Split LLM-eligible records ---
    meeting_records = [r for r in llm_eligible_records if r.get('category') == 'Meeting Announcement']
    other_records = [r for r in llm_eligible_records if r.get('category') != 'Meeting Announcement']
    app_logger.info(f"Found {len(meeting_records)} meeting announcements and {len(other_records)} other items for LLM processing.")

    # --- Cost/Call Warning ---
    # Calculate expected API usage based on content-based batching
    num_flash_calls_initial = len(meeting_records) + len(other_records)
    num_pro_calls_final = 2 # (1 for top stories, 1 for final report)
    
    app_logger.info(f"This operation will require approximately:")
    app_logger.info(f"  - {num_flash_calls_initial} initial calls to {config.flash_model_name} (for initial processing)")
    app_logger.info(f"  - {num_pro_calls_final} calls to {config.pro_model_name} (for final high-accuracy processing)")
    
    # More conservative check for low-RPM models
    expected_pro_minutes = num_pro_calls_final / config.pro_rpm
    expected_flash_minutes = num_flash_calls_initial / config.flash_rpm
    
    app_logger.info(f"  - Estimated time: ~{expected_pro_minutes:.1f} min for Pro model calls")
    app_logger.info(f"  - Estimated time: ~{expected_flash_minutes:.1f} min for Flash model calls")
    
    if num_pro_calls_final > 10:  # More conservative limit for 2 RPM model
        proceed = input("This will use the low-RPM Pro model multiple times. Proceed? (y/n): ").lower().strip()
        if proceed != 'y':
            app_logger.info("Operation cancelled by user.")
            return
            
    # --- Initialize Provider ---
    try:
        provider = get_provider(config, prompts)
    except Exception as e:
        app_logger.critical(f"Failed to initialize LLMProvider: {e}")
        return
        
    # --- Pipeline Step 1 & 2: Parse Meetings and Summarize News (Parallel Batches) ---
    app_logger.info("--- Starting Step 1: Parsing Meetings (Flash) ---")
    meetings_json = provider.parse_meeting_batch(meeting_records, current_date_str)
    
    app_logger.info("--- Starting Step 2: Summarizing News (Flash) ---")
    all_summaries = provider.summarize_news_batch(other_records, current_date_str)

    if not all_summaries and not meetings_json and not unpublished_records:
        app_logger.error("Failed to get any LLM-processed data, and no unpublished records found. Aborting.")
        return

    # --- Pipeline Step 3: Select Top Stories (Pro) ---
    app_logger.info("--- Starting Step 3: Selecting Top Stories (Pro) ---")
    top_story_urls = provider.select_top_stories(all_summaries, current_date_str)
    
    # --- Pipeline Step 4: Format Final Briefing (Pro) ---
    app_logger.info("--- Starting Step 4: Formatting Final Briefing (Pro) ---")
    final_briefing = provider.format_final_briefing(
        target_date=args.date,
        current_date=current_date_str,
        meetings_json=meetings_json,
        all_summaries=all_summaries,
        top_story_urls=top_story_urls
    )

    # --- NEW LOGIC: Append Unpublished Records Section ---
    if final_briefing and unpublished_records:
        app_logger.info("Appending unpublished pages section to briefing.")
        
        # Build the new section
        # NOTE: The requirement is for these to be marked as "new" and page text is "--unpublished".
        # We present the original URL and category from the database entry.
        unpublished_section = "\n\n## Newly unpublished pages\n\nThese pages, which were previously identified as new or updated, have been marked as unpublished by the source system (mass.gov returned the content as `--unpublished`).\n"
        
        for record in unpublished_records:
            # Check if the URL key exists before attempting to access it
            url = record.get('url', 'N/A URL')
            category = record.get('category', 'N/A Category')
            
            # Format as: * [Category] [URL]
            unpublished_section += f"\n* **{category}**: {url}"
            
        final_briefing += unpublished_section
        
    # --- Write Final Report ---
    # Resolve output dir relative to ROOT_DIR
    output_path = Path(args.output_dir)
    if not output_path.is_absolute():
        output_path = ROOT_DIR / output_path
    
    output_dir = output_path
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if final_briefing:
        briefing_filename = output_dir / f"briefing_{args.date}.md"
        with open(briefing_filename, 'w', encoding='utf-8') as f:
            f.write(final_briefing)
        app_logger.info(f"Successfully saved final news briefing to {briefing_filename}")
    else:
        app_logger.error("Failed to generate the final news briefing.")

    # --- Save raw/intermediate data for debugging (Append unpublished data for completeness) ---
    raw_filename = output_dir / f"briefing_{args.date}_intermediate_data.json"
    intermediate_data = {
        "parsed_meetings": meetings_json,
        "summarized_news": all_summaries,
        "selected_top_stories": top_story_urls,
        "unpublished_pages": [{"url": r.get('url'), "category": r.get('category')} for r in unpublished_records]
    }
    with open(raw_filename, 'w', encoding='utf-8') as f:
        json.dump(intermediate_data, f, indent=2)
    app_logger.info(f"Successfully saved intermediate data to {raw_filename}")

    app_logger.info("--- News Briefing Generation Complete ---")

if __name__ == "__main__":
    main()

