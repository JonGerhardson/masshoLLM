import yaml
import logging
import pandas as pd
import time
import random
import argparse
import os
from datetime import datetime, timedelta

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
        log_filename = os.path.join(log_dir, f"retry_run_on_{timestamp}_for_{csv_date_str}.log")
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

def process_url(session, url, lastmodified, recency_threshold):
    """Processes a single URL, determining its type and extracting data."""
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
            page_date_obj = scraper.find_best_date_on_page(html)
            if page_date_obj:
                record_data['page_date'] = page_date_obj.strftime("%Y-%m-%d")
                if datetime.now() - page_date_obj <= timedelta(days=recency_threshold):
                    record_data['is_new'] = 'yes'
                    logging.info("New article found based on page date.")
            else:
                record_data['is_new'] = 'maybe'
                logging.info("Page has no discernible date. Flagging as 'maybe'.")
            
            if record_data['is_new'] in ['yes', 'maybe']:
                content_for_llm = scraper.extract_article_content(html)
    
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
    parser = argparse.ArgumentParser(description="Scrape and analyze mass.gov for news leads.")
    parser.add_argument("--date", help="Specify a date to process in YYYY-MM-DD format. Defaults to yesterday.")
    parser.add_argument("--test", action="store_true", help="Run in test mode: process first 25 URLs and use testing.db.")
    parser.add_argument("--retry", nargs='?', const=True, default=False, 
                        help="Run in scraping retry mode.")
    # --- UPGRADE: Added new argument for the LLM retry feature. ---
    parser.add_argument("--retry_llm", action="store_true", help="Re-run LLM analysis on records that previously failed.")
    args = parser.parse_args()
    
    config = load_config()
    db_config = config['database_settings']
    agent_config = config['agent_settings']
    
    if args.date:
        try:
            datetime.strptime(args.date, "%Y-%m-%d")
            csv_date_str = args.date
        except ValueError:
            logging.critical(f"Invalid date format: '{args.date}'. Please use YYYY-MM-DD.")
            return
    else:
        if args.retry is True or args.retry_llm:
            logging.critical("Retry modes require a specific date. Please use the --date flag.")
            return
        yesterday = datetime.now() - timedelta(days=1)
        csv_date_str = yesterday.strftime("%Y-%m-%d")

    is_manual_retry = (args.retry is True)
    setup_main_logging(args.test, csv_date_str, is_manual_retry, args.retry_llm)
    
    table_name = f"massgov_{csv_date_str.replace('-', '_')}"
    db_filename = "testing.db" if args.test else db_config['database_file']
    llm_handler.setup_llm_logging(csv_date_str)
    conn = database.create_connection(db_filename)
    if not conn: return
    database.create_daily_table(conn, table_name)
    
    # --- UPGRADE: New execution path for the --retry_llm feature. ---
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
        
        summary_results = llm_handler.get_batch_summaries(llm_batch)
        
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

    # --- Standard Execution Path (Scraping) ---
    logging.info("--- Starting Mass.gov News Lead Agent ---")
    if args.test: logging.info("--- RUNNING IN TEST MODE ---")
    if is_manual_retry: logging.info("--- RUNNING IN AGGRESSIVE RETRY MODE ---")

    scraping_session = scraper.create_session_with_retries()
    llm_batch = []
    processed_records = []
    recency_threshold = agent_config.get('recency_threshold_days', 2)

    if is_manual_retry:
        retry_log_file = os.path.join('logs', f"{csv_date_str}_retries.log")
        # (This mode processes sequentially and is less likely to hit TPM limits, so we keep the end-of-run batch)
        # ... retry logic ...
    else: # Normal or Test run
        csv_url = agent_config['github_csv_url_format'].format(date=csv_date_str)
        try:
            daily_df = pd.read_csv(csv_url, names=['loc', 'lastmod'], header=0)
            if args.test: daily_df = daily_df.head(25)
            logging.info(f"Loaded {len(daily_df)} URLs for processing.")
        except Exception as e:
            logging.error(f"Failed to fetch or read CSV from {csv_url}. Error: {e}")
            if conn: conn.close()
            return
            
        # --- UPGRADE: Logic for incremental batching. ---
        total_urls = len(daily_df)
        # Define checkpoints at 25%, 50%, and 75%
        checkpoints = {int(total_urls * p) for p in [0.25, 0.5, 0.75]}

        for index, row in daily_df.iterrows():
            url, lastmodified = str(row['loc']).strip(), str(row['lastmod']).strip()

            logging.info(f"Processing URL ({index + 1}/{total_urls}): {url}")
            if database.check_if_url_exists(conn, table_name, url):
                logging.warning(f"URL already processed. Skipping.")
                continue

            try:
                time.sleep(random.uniform(agent_config['rate_limit_min'], agent_config['rate_limit_max']))
                record_data, content_for_llm = process_url(scraping_session, url, lastmodified, recency_threshold)
                if content_for_llm:
                    llm_batch.append({'url': url, 'content': content_for_llm, 'is_maybe': record_data['is_new'] == 'maybe'})
                processed_records.append(record_data)

            except scraper.Scraper403Error:
                logging.error(f"403 Forbidden error for {url}. Logging for retry.")
                log_retry_url(url, csv_date_str)
            except Exception as e:
                logging.error(f"An unexpected error occurred while processing {url}: {e}", exc_info=True)

            # --- UPGRADE: Check if we have reached a progress checkpoint. ---
            if (index + 1) in checkpoints:
                submit_llm_batch(llm_batch, processed_records)

    # --- UPGRADE: Process the final batch after the loop completes. ---
    # This handles the last 25% of items and any remaining items in manual retry mode.
    submit_llm_batch(llm_batch, processed_records)

    # (The logic for merging results is now inside submit_llm_batch, so we remove it from here)
    
    for record in processed_records:
        database.insert_record(conn, table_name, record)
    logging.info(f"Completed processing. {len(processed_records)} records saved to the database.")

    if not args.test:
        report_records = database.fetch_new_and_maybe_records(conn, table_name)
        if report_records:
            report_generator.generate_report(report_records, csv_date_str)
        else:
            logging.warning("No new or potentially new items were found to generate a report.")
    else:
        logging.info("Skipping report generation in test mode.")

    if conn: conn.close()
    logging.info("Database connection closed.")
    logging.info("--- Agent run finished successfully. ---")

if __name__ == "__main__":
    main()

