import logging
import argparse
import yaml
import os
import re
from typing import Optional
from pathlib import Path
from datetime import datetime

# --- Custom Modules ---
import database
from llm_providers import get_provider, BaseProvider

def setup_briefing_logging(date_str: str):
    """Configures logging for the briefing generator, including a detailed LLM log."""
    log_dir = Path("report_extras_logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_log_filename = f"{timestamp}_{date_str.replace('-', '')}"
    
    # --- Main Application Logger ---
    app_log_file = log_dir / f"{base_log_filename}.log"
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

    # --- Detailed LLM Logger ---
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

def main():
    parser = argparse.ArgumentParser(
        description="Generate a news briefing from a daily report.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("date", help="The date of the report to process, in YYYY-MM-DD format.")
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory for generated files.")
    parser.add_argument("--include-maybe", action="store_true", help="Include pages marked as 'maybe' in the LLM processing.")
    parser.add_argument("--lmstudio", action="store_true", help="Use LM Studio configuration from config.yaml for API calls.")
    args = parser.parse_args()
    
    if not re.match(r'\d{4}-\d{2}-\d{2}', args.date):
        print(f"CRITICAL: Invalid date format: '{args.date}'. Please use YYYY-MM-DD.")
        return

    app_logger, llm_logger = setup_briefing_logging(args.date)
    
    # --- Load Config ---
    try:
        with open("config.yaml", 'r') as f:
            full_config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        app_logger.critical(f"Could not load config.yaml: {e}")
        return

    # --- Database Retrieval ---
    db_file = full_config.get('database_settings', {}).get('database_file')
    if not db_file:
        app_logger.critical("Database file not specified in config.yaml")
        return

    conn = database.create_connection(db_file)
    if not conn: return
    
    table_name = f"massgov_{args.date.replace('-', '_')}"
    if args.include_maybe:
        app_logger.info(f"Fetching 'new' and 'maybe' records from table '{table_name}' (excluding marked pages)...")
        records_to_process = database.fetch_new_and_maybe_records_with_exclusions(conn, table_name)
    else:
        app_logger.info(f"Fetching only 'new' records from table '{table_name}' (excluding marked pages)...")
        records_to_process = database.fetch_new_records_with_exclusions(conn, table_name)
    conn.close()
    
    if not records_to_process:
        app_logger.warning(f"No new items found in the database for {args.date}. Cannot generate briefing.")
        return

    # --- Provider Selection ---
    provider_name = 'lmstudio' if args.lmstudio else 'gemini'
    try:
        llm_provider: Optional[BaseProvider] = get_provider(
            provider_name, 
            config=full_config,
            app_logger=app_logger,
            llm_logger=llm_logger
        )
        if not llm_provider:
             app_logger.critical("Failed to initialize LLM provider.")
             return
    except ValueError as e:
        app_logger.critical(e)
        return

    app_logger.info(f"--- Starting News Briefing Generation for {args.date} using {provider_name} ---")

    # --- LLM Processing ---
    summaries = llm_provider.get_summaries_in_batches(records_to_process)
    
    # --- Content Aggregation ---
    briefing_parts = []
    for record in records_to_process:
        url, summary_text = record.get('url'), summaries.get(record.get('url'))
        if summary_text and "API Error" not in summary_text:
            briefing_parts.append(f"[SOURCE: {url}]\n{summary_text}")
        else:
            app_logger.error(f"Failed to get a valid summary for URL {url}: {summary_text or 'Not found.'}")
    
    if not briefing_parts:
        app_logger.error("Failed to generate any valid briefing parts from the API.")
        return

    combined_content = "\n\n---\n\n".join(briefing_parts)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    raw_filename = output_dir / f"briefing_{args.date}_full_raw.md"
    
    with open(raw_filename, 'w', encoding='utf-8') as f: f.write(combined_content)
    app_logger.info(f"Successfully saved raw briefing content to {raw_filename}")

    # --- Final Formatting ---
    final_briefing = llm_provider.format_final_briefing(combined_content)
    
    # --- Write Final Report ---
    if final_briefing:
        briefing_filename = output_dir / f"briefing_{args.date}.md"
        with open(briefing_filename, 'w', encoding='utf-8') as f: f.write(final_briefing)
        app_logger.info(f"Successfully saved final news briefing to {briefing_filename}")
    else:
        app_logger.error("Failed to generate the final news briefing.")

    app_logger.info("--- News Briefing Generation Complete ---")

if __name__ == "__main__":
    main()


