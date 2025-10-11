# database.py
import sqlite3
from sqlite3 import Error
import logging
from datetime import datetime
from typing import Optional, Dict, Any

# --- Logging Setup ---
# It's good practice to log database operations and potential errors.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _sanitize_table_name(table_name: str) -> str:
    """
    Sanitizes a string to be a safe SQL table name to prevent SQL injection.
    A simple approach is to replace dashes with underscores and ensure it's a valid identifier.
    """
    return "".join(c if c.isalnum() or c == '_' else '_' for c in table_name)

def create_connection(db_file: str) -> Optional[sqlite3.Connection]:
    """
    Create a database connection to the SQLite database specified by db_file.
    
    Args:
        db_file (str): The path to the SQLite database file.
        
    Returns:
        Optional[sqlite3.Connection]: Connection object or None if an error occurs.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        logging.info(f"Successfully connected to SQLite database: {db_file}")
        return conn
    except Error as e:
        logging.error(f"Error connecting to database {db_file}: {e}")
        return None

def create_daily_table(conn: sqlite3.Connection, table_name: str) -> None:
    """
    Create a table for a specific day's data if it doesn't already exist.

    Args:
        conn (sqlite3.Connection): The database connection object.
        table_name (str): The name for the new table (e.g., 'massgov_2025_10_11').
    """
    sanitized_table_name = _sanitize_table_name(table_name)

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {sanitized_table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL UNIQUE,
        lastmodified TEXT,
        filetype TEXT,
        page_date TEXT,
        is_new TEXT CHECK(is_new IN ('yes', 'no', 'maybe')),
        summary TEXT,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
        logging.info(f"Table '{sanitized_table_name}' is ready.")
    except Error as e:
        logging.error(f"Error creating table '{sanitized_table_name}': {e}")

def check_if_url_exists(conn: sqlite3.Connection, table_name: str, url: str) -> bool:
    """
    Check if a URL has already been processed and inserted into the daily table.
    
    Args:
        conn (sqlite3.Connection): The database connection object.
        table_name (str): The name of the table to check.
        url (str): The URL to look for.
        
    Returns:
        bool: True if the URL exists, False otherwise.
    """
    sanitized_table_name = _sanitize_table_name(table_name)
    sql = f"SELECT id FROM {sanitized_table_name} WHERE url = ?"
    
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (url,))
        data = cursor.fetchone()
        return data is not None
    except Error as e:
        logging.error(f"Error checking URL '{url}' in table '{sanitized_table_name}': {e}")
        return False # Assume it doesn't exist if an error occurs to allow processing attempt.

def insert_record(conn: sqlite3.Connection, table_name: str, record_data: Dict[str, Any]) -> Optional[int]:
    """
    Insert a new record into the specified daily table.
    
    Args:
        conn (sqlite3.Connection): The database connection object.
        table_name (str): The name of the table to insert into.
        record_data (Dict[str, Any]): A dictionary containing the data for the new record.
                                     Keys should match the column names.
    
    Returns:
        Optional[int]: The ID of the newly inserted row, or None if insertion fails.
    """
    sanitized_table_name = _sanitize_table_name(table_name)
    
    sql = f''' INSERT INTO {sanitized_table_name}(url, lastmodified, filetype, page_date, is_new, summary)
               VALUES(?,?,?,?,?,?) '''
               
    # Prepare tuple of values in the correct order for the SQL statement
    values = (
        record_data.get('url'),
        record_data.get('lastmodified'),
        record_data.get('filetype'),
        record_data.get('page_date'),
        record_data.get('is_new'),
        record_data.get('summary')
    )

    try:
        cursor = conn.cursor()
        cursor.execute(sql, values)
        conn.commit()
        logging.info(f"Successfully inserted record for URL: {record_data.get('url')}")
        return cursor.lastrowid
    except Error as e:
        logging.error(f"Error inserting record for URL '{record_data.get('url')}': {e}")
        return None

# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    db_file = "massgov_updates_test.db"
    
    # Get today's date to generate the table name, e.g., 'massgov_2025_10_11'
    today_str = datetime.now().strftime("%Y_%m_%d")
    table_name = f"massgov_{today_str}"

    # 1. Create a database connection
    connection = create_connection(db_file)

    if connection:
        # 2. Create the daily table
        create_daily_table(connection, table_name)

        # 3. Example record to insert
        example_record = {
            'url': 'https://www.mass.gov/doc/test-document/download',
            'lastmodified': '2025-10-10T12:00:00Z',
            'filetype': 'PDF',
            'page_date': '2025-10-10',
            'is_new': 'yes',
            'summary': 'This is a test summary of the document.'
        }
        
        # 4. Check if the record already exists before inserting
        if not check_if_url_exists(connection, table_name, example_record['url']):
            # 5. Insert the record
            insert_record(connection, table_name, example_record)
        else:
            print(f"URL '{example_record['url']}' already exists in the database. Skipping.")

        # Close the connection
        connection.close()
        print("Database operations complete and connection closed.")


