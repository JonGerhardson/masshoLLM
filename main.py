import yaml
import logging
import pandas as pd
from datetime import datetime, timedelta
import time
import random
import argparse

# Import the custom modules we built
import database
import scraper
import llm_handler

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent_run.log"),
        logging.StreamHandler()
    ]
)

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

def main():
    """Main function to run the AI agent."""
    parser = argparse.ArgumentParser(description="Mass.gov News Lead Agent: Scrapes and analyzes daily updates.")
    parser.add_argument(
        "--date",
        help="The date of the CSV to process in YYYY-MM-DD format. If not provided, it defaults to yesterday."
    )
    args = parser.parse_args()

    logging.info("--- Starting Mass.gov News Lead Agent ---")
    
    config = load_config()
    db_config = config['database_settings']
    agent_config = config['agent_settings']
    rate_limit_config = agent_config.get('rate_limiting', {'min_delay_seconds': 1, 'max_delay_seconds': 3})

    # --- 1. Determine Processing Date and Generate URLs/Table Name ---
    today = datetime.now()
    
    if args.date:
        try:
            # Use the date provided via the command-line argument
            processing_date = datetime.strptime(args.date, "%Y-%m-%d")
            csv_date_str = args.date
            logging.info(f"Using manually specified date from command line: {csv_date_str}")
        except ValueError:
            logging.critical(f"Invalid date format for --date argument: '{args.date}'. Please use YYYY-MM-DD.")
            exit()
    else:
        # Default to yesterday if no date is provided
        yesterday = today - timedelta(days=1)
        csv_date_str = yesterday.strftime("%Y-%m-%d")
        logging.info(f"No date specified. Defaulting to yesterday's data: {csv_date_str}")

    csv_url = agent_config['github_csv_url_format'].format(date=csv_date_str)
    table_name = f"massgov_{csv_date_str.replace('-', '_')}"
    
    logging.info(f"Generated daily CSV URL for date: {csv_date_str}")

    # --- 2. Initialize Database ---
    conn = database.create_connection(db_config['database_file'])
    if not conn:
        return
        
    database.create_daily_table(conn, table_name)
    logging.info(f"Table '{table_name}' is ready.")

    # --- 3. Fetch and Load CSV ---
    try:
        daily_df = pd.read_csv(csv_url, names=['loc', 'lastmod'], header=0)
        logging.info(f"Successfully loaded CSV with {len(daily_df)} rows.")
    except Exception as e:
        logging.error(f"Failed to fetch or read CSV from {csv_url}. Error: {e}")
        conn.close()
        return

    # --- 4. Create scraping session and prepare for batching ---
    scraping_session = scraper.create_session_with_retries()
    all_records = []
    summarization_batch = []

    # --- 5. Main Processing Loop (Content Gathering) ---
    for index, row in daily_df.iterrows():
        url = row['loc'].strip()
        lastmodified = row['lastmod']
        
        logging.info(f"Processing URL ({index + 1}/{len(daily_df)}): {url}")

        if database.check_if_url_exists(conn, table_name, url):
            logging.warning(f"URL already processed. Skipping.")
            continue

        record_data = {
            'url': url, 'lastmodified': lastmodified, 'filetype': None,
            'page_date': None, 'is_new': 'no', 'summary': None
        }

        try:
            if "/download" in url:
                logging.info("URL identified as a document download.")
                landing_page_url = url.rsplit('/download', 1)[0]
                html = scraper.fetch_page(scraping_session, landing_page_url)
                if html:
                    page_date_obj = scraper.extract_download_page_date(html)
                    if page_date_obj:
                        record_data['page_date'] = page_date_obj.strftime("%Y-%m-%d")
                        if today - page_date_obj <= timedelta(days=agent_config['recency_threshold_days']):
                            record_data['is_new'] = 'yes'
                            logging.info("New document found. Queueing for summarization.")
                            doc_content, doc_type = scraper.download_and_extract_document_text(scraping_session, url)
                            record_data['filetype'] = doc_type
                            if doc_content:
                                summarization_batch.append({'url': url, 'content': doc_content, 'is_maybe': False})
                            else:
                                logging.warning("Could not extract text from document.")
            else:
                logging.info("URL identified as a standard page.")
                record_data['filetype'] = 'HTML'
                html = scraper.fetch_page(scraping_session, url)
                if html:
                    page_date_obj = scraper.find_best_date_on_page(html)
                    if page_date_obj:
                        record_data['page_date'] = page_date_obj.strftime("%Y-%m-%d")
                        record_data['is_new'] = 'yes' if today - page_date_obj <= timedelta(days=agent_config['recency_threshold_days']) else 'no'
                    else:
                        record_data['is_new'] = 'maybe'
                    
                    if record_data['is_new'] in ['yes', 'maybe']:
                        logging.info("Page is new or of unknown age. Queueing for summarization.")
                        main_content = scraper.extract_article_content(html)
                        if main_content:
                            summarization_batch.append({'url': url, 'content': main_content, 'is_maybe': (record_data['is_new'] == 'maybe')})
                        else:
                            logging.warning("Could not extract main content.")
            
            all_records.append(record_data)

        except Exception as e:
            logging.error(f"An unexpected error occurred while processing {url}: {e}", exc_info=True)
            continue
        
        delay = random.uniform(rate_limit_config['min_delay_seconds'], rate_limit_config['max_delay_seconds'])
        logging.info(f"Waiting for {delay:.2f} seconds.")
        time.sleep(delay)

    # --- 6. Batch Summarization ---
    if summarization_batch:
        logging.info(f"Starting batch summarization for {len(summarization_batch)} documents.")
        batch_summaries = llm_handler.get_batch_summaries(summarization_batch)
        
        for record in all_records:
            if record['url'] in batch_summaries:
                record['summary'] = batch_summaries[record['url']]

    # --- 7. Database Insertion ---
    logging.info(f"Inserting {len(all_records)} records into the database.")
    for record in all_records:
        database.insert_record(conn, table_name, record)
        logging.info(f"Successfully inserted record for URL: {record['url']}")

    # --- 8. Clean Up ---
    conn.close()
    logging.info("Database connection closed.")
    logging.info("--- Agent run finished successfully. ---")

if __name__ == "__main__":
    main()


