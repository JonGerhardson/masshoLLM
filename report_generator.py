import logging
import argparse
import yaml
from datetime import datetime
from typing import List, Dict, Any

# Import the database module to interact with the database
import database

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_report(records: List[Dict[str, Any]], csv_date_str: str):
    """
    Generates a Markdown report from a list of records, categorizing them
    based on the LLM's analysis.
    
    UPDATE: Now includes sections for "Meeting Announcements" and "Meeting Materials".
    """
    report_filename = f"report_{csv_date_str}.md"
    logging.info(f"Generating daily report: {report_filename}")

    # --- Categorize Records ---
    new_announcements = []
    press_releases = []
    meeting_announcements = [] # --- NEW ---
    meeting_materials = [] # --- NEW ---
    new_documents = []
    might_be_new = []

    # --- UPDATED LOGIC ---
    # This now correctly reads the category field directly from the database record.
    for record in records:
        category = record.get('category')

        if category == 'New Announcement':
            new_announcements.append(record)
        elif category == 'Press Release':
            press_releases.append(record)
        elif category == 'Meeting Announcement': # --- NEW ---
            meeting_announcements.append(record)
        elif category == 'Meeting Materials': # --- NEW ---
            meeting_materials.append(record)
        elif category == 'New Document':
            new_documents.append(record)
        elif category == 'Recent Update':
            might_be_new.append(record)
        # All other categories ('Timeless Info', 'API Error', etc.) are ignored for the report.

    # --- Format Date ---
    try:
        date_obj = datetime.strptime(csv_date_str, '%Y-%m-%d')
        day = date_obj.day
        formatted_date = f"{date_obj.strftime('%B')} {day}, {date_obj.year}"
    except (ValueError, TypeError):
        formatted_date = csv_date_str

    # --- Build the Markdown Report ---
    markdown_content = f"# The Commonwealth Log File\n\n"
    markdown_content += f"**{formatted_date}**\n\n"
    markdown_content += "---\n\n"

    # Section 1: New Announcements
    markdown_content += "## New announcements\n\n"
    if new_announcements:
        for item in new_announcements:
            markdown_content += f"### [{item['url']}]({item['url']})\n"
            markdown_content += f"**Summary:** {item.get('summary', 'No summary generated.')}\n\n"
    else:
        markdown_content += "_No new announcements were identified._\n\n"

    # Section 2: Press Releases
    markdown_content += "## Press Releases\n\n"
    if press_releases:
        for item in press_releases:
            markdown_content += f"### [{item['url']}]({item['url']})\n"
            markdown_content += f"**Summary:** {item.get('summary', 'No summary generated.')}\n\n"
    else:
        markdown_content += "_No new press releases were identified._\n\n"

    # --- NEW Section 3: Meeting Announcements ---
    markdown_content += "## Meeting announcements\n\n"
    if meeting_announcements:
        for item in meeting_announcements:
            markdown_content += f"### [{item['url']}]({item['url']})\n"
            markdown_content += f"**Meeting Date:** {item.get('page_date', 'N/A')}\n"
            markdown_content += f"**Summary:** {item.get('summary', 'No summary generated.')}\n\n"
    else:
        markdown_content += "_No new meeting announcements were identified._\n\n"

    # --- NEW Section 4: Meeting Materials ---
    markdown_content += "## Meeting materials\n\n"
    if meeting_materials:
        for item in meeting_materials:
            markdown_content += f"### [{item['url']}]({item['url']})\n"
            markdown_content += f"**Summary:** {item.get('summary', 'No summary generated.')}\n\n"
    else:
        markdown_content += "_No new meeting materials were identified._\n\n"

    # Section 5: New Documents
    markdown_content += "## New documents\n\n"
    if new_documents:
        for item in new_documents:
            markdown_content += f"### [{item['url']}]({item['url']})\n"
            markdown_content += f"**Summary:** {item.get('summary', 'No summary generated.')}\n"
            markdown_content += f"**File Type:** {item.get('filetype', 'N/A')} | **Published Date:** {item.get('page_date', 'N/A')}\n\n"
    else:
        markdown_content += "_No new documents with confirmed dates were found._\n\n"

    # Section 6: Might Be New?
    markdown_content += "## Might be new?\n\n"
    if might_be_new:
        for item in might_be_new:
            markdown_content += f"### [{item['url']}]({item['url']})\n"
            markdown_content += f"**Summary:** {item.get('summary', 'No summary generated.')}\n\n"
    else:
        markdown_content += "_No items were identified as recent updates._\n\n"

    # --- Write Report to File ---
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        logging.info(f"Successfully saved report to {report_filename}")
    except IOError as e:
        logging.error(f"Failed to write report file: {e}")


def main():
    """
    Main function to run the report generator as a standalone script.
    """
    parser = argparse.ArgumentParser(description="Generate a Markdown report for a specific day's data.")
    parser.add_argument("date", help="The date to generate the report for, in YYYY-MM-DD format.")
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
        csv_date_str = args.date
    except ValueError:
        logging.critical(f"Invalid date format: '{args.date}'. Please use YYYY-MM-DD.")
        return

    logging.info(f"--- Standalone Report Generation for {csv_date_str} ---")

    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
            db_file = config['database_settings']['database_file']
    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        logging.critical(f"Could not load database configuration from config.yaml: {e}")
        return

    conn = database.create_connection(db_file)
    if not conn: return

    table_name = f"massgov_{csv_date_str.replace('-', '_')}"
    
    logging.info(f"Fetching records from table '{table_name}'...")
    report_records = database.fetch_new_and_maybe_records(conn, table_name)

    if report_records:
        generate_report(report_records, csv_date_str)
    else:
        logging.warning(f"No new or potentially new items were found in the database for {csv_date_str}.")

    if conn: conn.close()
    logging.info("--- Report generation complete. ---")


if __name__ == "__main__":
    main()
