import yaml
import logging
import pandas as pd
import time
import random
import argparse
import os
import dateparser
from datetime import datetime, timedelta
import requests # Import requests to handle exceptions
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify that API key is available (using the same key as llm_handler)
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("Warning: GEMINI_API_KEY not found in environment variables!")
    print("Please ensure your .env file contains the GEMINI_API_KEY variable.")
else:
    print("API key loaded successfully from .env file.")

# Import the custom modules we built
import database
import scraper
import llm_handler
import report_generator

def setup_main_logging(is_test_mode=False, csv_date_str=None, is_retry_mode=False, is_llm_retry_mode=False):
    """Configures the main application logger."""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_filename = os.path.join(log_dir, "agent_run.log")
    timestamp = datetime.now().strftime("%m%d%Y%H%M")
    if is_test_mode:
        log_filename = os.path.join(log_dir, f"test_on_{timestamp}_for_{csv_date_str}_csv.log")
    elif is_retry_mode:
        log_filename = os.path.join(log_dir, f"scraping_retry_run_on_{timestamp}_for_{csv_date_str}.log") # Updated log name
    elif is_llm_retry_mode:
        log_filename = os.path.join(log_dir, f"llm_retry_on_{timestamp}_for_{csv_date_str}.log")

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='a'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Main logging configured to file: {log_filename}")

def log_retry_url(url: str, date_str: str):
    """Appends a URL that resulted in a 403 error to a date-stamped log file."""
    retry_log_filename = os.path.join('logs', f"{date_str}_retries.log")
    try:
        with open(retry_log_filename, 'a') as f:
            f.write(f"{url}\n")
    except IOError as e:
        logging.error(f"Could not write to retry log file {retry_log_filename}: {e}")

def log_failed_retry_session(urls_to_save, csv_date_str):
    """Saves unprocessed URLs from a failed retry session."""
    timestamp = datetime.now().strftime("%m%d%Y%H%M")
    failed_log_path = os.path.join('logs', f"FAILED_RETRY_ON_{timestamp}_for_{csv_date_str}.log")
    try:
        with open(failed_log_path, 'w') as f:
            for url in urls_to_save:
                f.write(f"{url}\n")
        logging.critical(f"Gracefully aborted. Remaining URLs saved to {failed_log_path}")
    except IOError as e:
        logging.critical(f"Could not write to FAILED_RETRY log file: {e}")

def load_config():
    """Loads the main configuration file."""
    try:
        with open("config.yaml", 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.critical("CRITICAL ERROR: config.yaml not found. The application cannot run.")
        exit()
    except yaml.YAMLError as e:
        logging.critical(f"CRITICAL ERROR: Error parsing config.yaml: {e}")
        exit()

def process_url(session, url, lastmodified, recency_threshold, target_date_obj):
    """
    Processes a single URL, determining its type and extracting data.
    """
    record_data = {
        'url': url,
        'lastmodified': lastmodified,
        'is_new': 'no',
        'summary': None,
        'filetype': None,
        'page_date': None,
        'category': None,
        'extracted_text': None # Initialize new field
    }
    content_for_llm = None

    try:
        if "/download" in url:
            logging.info("URL identified as a document download. Using document workflow.")
            
            if url.endswith('/download/'):
                landing_page_url = url[:-10]
            elif url.endswith('/download'):
                landing_page_url = url[:-9]
            else:
                landing_page_url = url.split('/download')[0]

            html = scraper.fetch_page(session, landing_page_url)
            
            if html:
                page_date_obj = scraper.extract_download_page_date(html)
                if page_date_obj:
                    record_data['page_date'] = page_date_obj.strftime("%Y-%m-%d")
                    if datetime.now() - page_date_obj <= timedelta(days=recency_threshold):
                        record_data['is_new'] = 'yes'
            
            if record_data['is_new'] == 'yes':
                  logging.info("New document found. Proceeding to download and analyze file type.")
                  doc_data = scraper.download_and_extract_document_text(session, url)
                  record_data['filetype'] = doc_data.get("filetype")
                  content_for_llm = doc_data.get("content_for_llm")
        else:
            logging.info("URL identified as a standard page. Using article workflow.")
            record_data['filetype'] = 'HTML'
            html = scraper.fetch_page(session, url)
            
            if html:
                # --- UPDATED DATE LOGIC ---
                date_info = scraper.find_best_date_on_page(html)
                posting_date = date_info.get('posting_date')
                meeting_date = date_info.get('meeting_date')
                
                # Get yesterday's date, respecting the target_date we are running for
                yesterday = (target_date_obj - timedelta(days=1)).date()

                if meeting_date:
                    logging.info(f"Meeting date found: {meeting_date.strftime('%Y-%m-%d')}")
                    record_data['page_date'] = meeting_date.strftime("%Y-%m-%d")
                    # Check if meeting is yesterday, today, or in the future
                    if meeting_date.date() >= yesterday:
                        record_data['is_new'] = 'yes'
                        logging.info("New meeting announcement found based on event date.")
                    else:
                        record_data['is_new'] = 'maybe' # It's a past meeting
                        logging.info("Past meeting page found. Flagging as 'maybe' for LLM classification (Meeting Materials).")
                
                elif posting_date:
                    logging.info(f"Posting date found: {posting_date.strftime('%Y-%m-%d')}")
                    record_data['page_date'] = posting_date.strftime("%Y-%m-%d")
                    # Check if posting date is within the recency threshold of *today*
                    if (datetime.now() - posting_date) <= timedelta(days=recency_threshold):
                        record_data['is_new'] = 'yes'
                        logging.info("New article found based on posting date.")
                    else:
                        record_data['is_new'] = 'maybe' # It's an old article
                
                else:
                    record_data['is_new'] = 'maybe'
                    logging.info("Page has no discernible date. Flagging as 'maybe'.")
                
                if record_data['is_new'] in ['yes', 'maybe']:
                    content_for_llm = scraper.extract_article_content(html)
            
    except requests.exceptions.HTTPError as e:
        if "404 Client Error" in str(e):
            # Check if the error message contains the pattern indicating an unpublished page
            if "---unpublished" in str(e):
                logging.warning(f"URL resulted in 404 Unpublished error: {url}. Recording status.")
                record_data['extracted_text'] = "PAGE STATUS: UNPUBLISHED/REMOVED"
                record_data['is_new'] = 'yes' # Treat as "new" deletion/change for reporting
                record_data['category'] = "Removed/Unpublished"
                return record_data, None # Return the record, no content for LLM
            else:
                logging.error(f"Error fetching URL {url}: {e}")
        else:
            raise # Re-raise all other HTTP errors (like 403)
    
    # --- UPGRADE: Store the extracted text in the record for potential retries. ---
    record_data['extracted_text'] = content_for_llm
    return record_data, content_for_llm

# --- NEW: Helper function to submit LLM batches incrementally. ---
def submit_llm_batch(llm_batch: list, processed_records: list):
    """Submits the current batch to the LLM and merges results."""
    if not llm_batch:
        return
    
    logging.info(f"Progress checkpoint reached. Submitting a batch of {len(llm_batch)} items to the LLM.")
    summary_results = llm_handler.get_batch_summaries(llm_batch)
    
    # Create a map of URL to record for efficient updating of the main list
    # This ensures we only search through the relevant slice of processed_records
    urls_in_batch = {item['url'] for item in llm_batch}
    url_to_record_map = {rec['url']: rec for rec in processed_records if rec['url'] in urls_in_batch}

    for url, result in summary_results.items():
        if url in url_to_record_map:
            url_to_record_map[url]['category'] = result.get('category')
            url_to_record_map[url]['summary'] = result.get('summary')
    
    llm_batch.clear()
    logging.info("Batch processing complete. Continuing scraping.")

def main():
    """Main function to run the AI agent."""
    parser = argparse.ArgumentParser(
        description="A command-line agent to scrape, analyze, and report on updates from mass.gov.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples of use:
  - Run for yesterday's date:
    python main.py

  - Run for a specific date:
    python main.py --date 2025-10-14

  - Run a quick test for a specific date's sitemap:
    python main.py --date 2025-10-14 --test

  - **NEW: Retry scraping for missing content:**
    python main.py --date 2025-10-14 --retry

  - Re-run LLM analysis on records that previously failed:
    python main.py --date 2025-10-14 --retry-llm

  - Update an existing run for a specific date with only the latest news API data:
    python main.py --date 2025-10-14 --news-only
"""
    )

    # Primary Operations
    primary_group = parser.add_argument_group('Primary Operations')
    primary_group.add_argument("--date", help="Specify a date to process in YYYY-MM-DD format. Defaults to yesterday.")

    # Special Modes
    mode_group = parser.add_argument_group('Special Modes (use one at a time with --date)')
    mode_group.add_argument("--test", action="store_true", help="Run in test mode: process first 25 URLs and use testing.db.")
    mode_group.add_argument("--retry", action="store_true", 
                        help="Run in scraping retry mode for URLs that are missing data in the 'extracted_text' column.")
    mode_group.add_argument("--retry_llm", action="store_true", help="Re-run LLM analysis on records that previously failed (e.g., due to API errors).")
    mode_group.add_argument("--news-only", action="store_true", help="Only fetch data from the /news API to update a day's run, skipping the sitemap.")
    
    args = parser.parse_args()
    
    config = load_config()
    db_config = config['database_settings']
    agent_config = config['agent_settings']
    
    if args.date:
        try:
            target_date_obj = datetime.strptime(args.date, "%Y-%m-%d")
            csv_date_str = args.date
        except ValueError:
            logging.critical(f"Invalid date format: '{args.date}'. Please use YYYY-MM-DD.")
            return
    else:
        if args.retry or args.retry_llm or args.news_only:
            logging.critical("Retry and news-only modes require a specific date. Please use the --date flag.")
            return
        yesterday = datetime.now() - timedelta(days=1)
        target_date_obj = yesterday
        csv_date_str = yesterday.strftime("%Y-%m-%d")

    # is_scraping_retry tracks the new, database-driven retry mode
    is_scraping_retry = args.retry 
    setup_main_logging(args.test, csv_date_str, is_scraping_retry, args.retry_llm)

    if args.news_only and (is_scraping_retry or args.retry_llm):
        logging.critical("--news-only cannot be used with --retry or --retry_llm modes.")
        return
    
    table_name = f"massgov_{csv_date_str.replace('-', '_')}"
    db_filename = "testing.db" if args.test else db_config['database_file']
    llm_handler.setup_llm_logging(csv_date_str)
    conn = database.create_connection(db_filename)
    if not conn: return
    database.create_daily_table(conn, table_name)
    
    # --- LLM-Retry Execution Path (Unchanged) ---
    if args.retry_llm:
        logging.info("--- Starting Agent in LLM-Retry Mode ---")
        records_to_retry = database.fetch_records_for_llm_retry(conn, table_name)
        if not records_to_retry:
            logging.info("No records found that require LLM reprocessing.")
            if conn: conn.close()
            return
        
        llm_batch = [
            {'url': r['url'], 'content': r['extracted_text'], 'is_maybe': r['is_new'] == 'maybe'}
            for r in records_to_retry
        ]
        
        summary_results = llm_handler.get_batch_summaries(llm_batch, db_connection=conn, table_name=table_name)
        
        # Create a map of URL -> ID for efficient updates
        url_to_id_map = {r['url']: r['id'] for r in records_to_retry}
        
        updated_count = 0
        for url, result in summary_results.items():
            record_id = url_to_id_map.get(url)
            if record_id and result.get('category') != "API Error":
                database.update_llm_results(conn, table_name, record_id, result['category'], result['summary'])
                logging.info(f"Successfully updated record for {url}")
                updated_count += 1
            else:
                logging.error(f"Could not update record for {url}, either ID was not found or LLM call failed again.")

        logging.info(f"LLM retry complete. Successfully updated {updated_count} records.")
        # Optional: Regenerate report after updating
        if updated_count > 0 and not args.test:
             report_records = database.fetch_new_and_maybe_records(conn, table_name)
             report_generator.generate_report(report_records, csv_date_str)

        if conn: conn.close()
        return # End execution for this mode

    # --- Standard/Scraping Retry Execution Path (Scraping) ---
    logging.info("--- Starting Mass.gov News Lead Agent ---")
    if args.test: logging.info("--- RUNNING IN TEST MODE ---")
    if is_scraping_retry: logging.info("--- RUNNING IN SCRAPING RETRY MODE (Missing Extracted Text) ---")
    if args.news_only: logging.info("--- RUNNING IN NEWS-ONLY MODE ---")


    scraping_session = scraper.create_session_with_retries()
    
    if not args.news_only:
        logging.info("--- Processing URLs from Sitemap ---")
        llm_batch = []
        # processed_records list is now only used for records inserted/updated in this run
        processed_records = [] 
        recency_threshold = agent_config.get('recency_threshold_days', 2)

        if is_scraping_retry:
            # --- NEW LOGIC: Fetch URLs from DB where extracted_text is NULL/'' ---
            daily_urls = database.fetch_urls_for_scraping_retry(conn, table_name)
            total_urls = len(daily_urls)
            if total_urls == 0:
                logging.info("Scraping retry mode finished: No URLs found with missing extracted text.")
                if conn: conn.close()
                return # Exit early
            logging.info(f"Loaded {total_urls} URLs for reprocessing from database.")
        else: # Normal or Test run
            csv_url = agent_config['github_csv_url_format'].format(date=csv_date_str)
            try:
                daily_df = pd.read_csv(csv_url, names=['loc', 'lastmod'], header=0)
                if args.test: daily_df = daily_df.head(25)
                # Convert DataFrame to a list of dicts for unified processing structure
                daily_urls = daily_df[['loc', 'lastmod']].rename(columns={'loc': 'url', 'lastmod': 'lastmodified'}).to_dict('records')
                total_urls = len(daily_urls)
                logging.info(f"Loaded {total_urls} URLs for processing from sitemap.")
            except Exception as e:
                logging.error(f"Failed to fetch or read CSV from {csv_url}. Error: {e}")
                if conn: conn.close()
                return
                
        # Define checkpoints at 25%, 50%, and 75% for incremental LLM submission
        checkpoints = {int(total_urls * p) for p in [0.25, 0.5, 0.75]}

        for index, row in enumerate(daily_urls):
            url = str(row['url']).strip()
            lastmodified = str(row['lastmodified']).strip()

            logging.info(f"Processing URL ({index + 1}/{total_urls}): {url}")
            
            # In normal mode, we skip if the URL is already processed.
            # In retry mode, we know the URL exists and is missing text, so we don't check.
            if not is_scraping_retry and database.check_if_url_exists(conn, table_name, url):
                logging.warning(f"URL already processed. Skipping.")
                continue

            try:
                time.sleep(random.uniform(agent_config['rate_limit_min'], agent_config['rate_limit_max']))
                
                record_data, content_for_llm = process_url(scraping_session, url, lastmodified, recency_threshold, target_date_obj)
                
                # --- MODIFIED: Skip LLM batching in scraping retry mode ---
                if not is_scraping_retry: 
                    if content_for_llm and record_data['is_new'] in ['yes', 'maybe']:
                        llm_batch.append({'url': url, 'content': content_for_llm, 'is_maybe': record_data['is_new'] == 'maybe'})
                
                # We always add the record to processed_records if it was scraped, even if extraction failed.
                processed_records.append(record_data)

            except scraper.Scraper403Error:
                logging.error(f"403 Forbidden error for {url}. Logging for retry in case DB update fails.")
                # Log to file, even though main --retry is now DB driven, to capture failed retries
                log_retry_url(url, csv_date_str) 
            except Exception as e:
                logging.error(f"An unexpected error occurred while processing {url}: {e}", exc_info=True)

            # Check if we have reached a progress checkpoint (only in non-retry mode).
            if not is_scraping_retry and (index + 1) in checkpoints:
                submit_llm_batch(llm_batch, processed_records)

        # Process the final batch after the loop completes (only in non-retry mode).
        if not is_scraping_retry:
            submit_llm_batch(llm_batch, processed_records)
        
        # Insert or Update the records
        for record in processed_records:
            if is_scraping_retry:
                # If in retry mode, update the existing record with new scraped data
                database.update_scraped_record(conn, table_name, record)
            else:
                # If in normal mode, attempt to insert
                database.insert_record(conn, table_name, record)
        
        logging.info(f"Completed processing sitemap URLs. {len(processed_records)} records saved/updated in the database.")
    else:
        logging.info("--- Skipping Sitemap Processing (--news-only flag detected) ---")

    # --- Process Press Releases from the News API (Unchanged) ---
    logging.info("--- Processing Press Releases from News API ---")
    press_releases = scraper.fetch_press_releases(scraping_session)
    if press_releases:
        one_day = timedelta(days=1)
        llm_batch_pr = []
        records_pr = []

        for item in press_releases:
            try:
                pub_date_str = item.get('datePublished')
                if not pub_date_str:
                    continue
                
                pub_date = dateparser.parse(pub_date_str)
                if not pub_date:
                    continue
                
                # CORRECTED: Filter against the target date of the run, not the current date
                if abs(target_date_obj.date() - pub_date.date()) > one_day:
                    continue

                # CORRECTED: The API provides a full URL, no need to build it
                full_url = item.get('url')
                if not full_url:
                    continue
                
                if database.check_if_url_exists(conn, table_name, full_url):
                    logging.info(f"Press release already exists in database. Skipping: {full_url}")
                    continue
                
                logging.info(f"Processing new press release: {full_url}")
                time.sleep(random.uniform(agent_config['rate_limit_min'], agent_config['rate_limit_max']))
                
                html = scraper.fetch_page(scraping_session, full_url)
                if not html:
                    logging.warning(f"Could not fetch HTML for press release: {full_url}")
                    continue
                
                content = scraper.extract_article_content(html)
                if not content:
                    logging.warning(f"Could not extract content for press release: {full_url}")
                    continue

                record = {
                    'url': full_url,
                    'lastmodified': pub_date.isoformat(),
                    'is_new': 'yes',
                    'summary': None,
                    'filetype': 'HTML',
                    'page_date': pub_date.strftime("%Y-%m-%d"),
                    'category': None,
                    'extracted_text': content
                }
                records_pr.append(record)
                # Give minor priority by classifying as 'maybe' for report_extras
                is_maybe = abs(target_date_obj.date() - pub_date.date()) <= one_day
                llm_batch_pr.append({'url': full_url, 'content': content, 'is_maybe': is_maybe})

            except Exception as e:
                logging.error(f"An unexpected error occurred while processing press release item {item.get('url')}: {e}", exc_info=True)
        
        # --- MODIFIED: Skip LLM processing for press releases in scraping retry mode ---
        if llm_batch_pr and not is_scraping_retry:
            logging.info(f"Submitting a batch of {len(llm_batch_pr)} press releases to the LLM for summary.")
            summary_results = llm_handler.get_batch_summaries(llm_batch_pr, db_connection=conn, table_name=table_name)
            
            url_to_record_map_pr = {rec['url']: rec for rec in records_pr}

            for url, result in summary_results.items():
                if url in url_to_record_map_pr:
                    # Manually set category to 'Press Release' as requested
                    url_to_record_map_pr[url]['category'] = "Press Release" 
                    url_to_record_map_pr[url]['summary'] = result.get('summary')
            
            for record in records_pr:
                database.insert_record(conn, table_name, record)
            logging.info(f"Completed processing press releases. {len(records_pr)} new records saved.")
        elif records_pr and is_scraping_retry:
             # In retry mode, we still save the newly scraped press releases, 
             # but without LLM results, to the DB for later LLM retry.
             for record in records_pr:
                database.insert_record(conn, table_name, record)
             logging.info(f"Completed processing press releases. {len(records_pr)} new records saved (LLM skipped).")


    # --- MODIFIED: Skip report generation when in scraping retry mode ---
    if not args.test and not is_scraping_retry:
        report_records = database.fetch_new_and_maybe_records(conn, table_name)
        if report_records:
            report_generator.generate_report(report_records, csv_date_str)
        else:
            logging.warning("No new or potentially new items were found to generate a report.")
    elif args.test:
        logging.info("Skipping report generation in test mode.")
    elif is_scraping_retry:
        logging.info("Skipping report generation in scraping retry mode.")

    if conn: conn.close()
    logging.info("Database connection closed.")
    logging.info("--- Agent run finished successfully. ---")

if __name__ == "__main__":
    main()
