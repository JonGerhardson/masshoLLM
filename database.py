import sqlite3
import logging
from typing import List, Dict, Any, Optional

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _sanitize_table_name(table_name: str) -> str:
    """Sanitizes a table name to prevent SQL injection."""
    return "".join(c for c in table_name if c.isalnum() or c == '_')

def create_connection(db_file: str) -> Optional[sqlite3.Connection]:
    """Create a database connection to a SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        logging.info(f"Successfully connected to SQLite database: {db_file}")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Error connecting to database: {e}")
    return None

def create_daily_table(conn: sqlite3.Connection, table_name: str):
    """Creates a new table for the daily run if it doesn't exist."""
    sanitized_table_name = _sanitize_table_name(table_name)
    # Schema now includes the 'extracted_text' column for the retry feature.
    sql_create_table = f"""
    CREATE TABLE IF NOT EXISTS {sanitized_table_name} (
        id INTEGER PRIMARY KEY,
        url TEXT NOT NULL UNIQUE,
        lastmodified TEXT,
        filetype TEXT,
        page_date TEXT,
        is_new TEXT,
        category TEXT,
        summary TEXT,
        extracted_text TEXT 
    );
    """
    try:
        c = conn.cursor()
        c.execute(sql_create_table)
    except sqlite3.Error as e:
        logging.error(f"Error creating table {sanitized_table_name}: {e}")

def check_if_url_exists(conn: sqlite3.Connection, table_name: str, url: str) -> bool:
    """Checks if a URL already exists in the specified table."""
    sanitized_table_name = _sanitize_table_name(table_name)
    sql = f'SELECT id FROM {sanitized_table_name} WHERE url = ?'
    cur = conn.cursor()
    cur.execute(sql, (url,))
    data = cur.fetchone()
    return data is not None

def insert_record(conn: sqlite3.Connection, table_name: str, record_data: Dict[str, Any]):
    """Inserts a single processed record into the database."""
    sanitized_table_name = _sanitize_table_name(table_name)
    # Updated SQL statement to include the new 'extracted_text' column.
    sql = f''' INSERT INTO {sanitized_table_name}(url, lastmodified, filetype, page_date, is_new, category, summary, extracted_text)
               VALUES(?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    try:
        cur.execute(sql, (
            record_data['url'],
            record_data['lastmodified'],
            record_data['filetype'],
            record_data['page_date'],
            record_data['is_new'],
            record_data.get('category'),
            record_data['summary'],
            record_data.get('extracted_text') # Add extracted_text to the insert tuple
        ))
        conn.commit()
    except sqlite3.IntegrityError:
        logging.warning(f"URL already exists in database, skipping insert: {record_data['url']}")
    except sqlite3.Error as e:
        logging.error(f"Error inserting record for URL {record_data['url']}: {e}")

def fetch_new_and_maybe_records(conn: sqlite3.Connection, table_name: str) -> List[Dict[str, Any]]:
    """Fetches all records marked as 'yes' or 'maybe' for reporting."""
    sanitized_table_name = _sanitize_table_name(table_name)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT * FROM {sanitized_table_name} WHERE is_new IN ('yes', 'maybe')")
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logging.error(f"Could not fetch records for report from table {sanitized_table_name}: {e}")
        return []

# --- NEW FUNCTION for --retry_llm feature ---
def fetch_records_for_llm_retry(conn: sqlite3.Connection, table_name: str) -> List[Dict[str, Any]]:
    """Fetches records that have stored text but failed LLM processing."""
    sanitized_table_name = _sanitize_table_name(table_name)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        # Fetches records where the text was extracted but the LLM failed (API Error) or wasn't run.
        sql = f"SELECT id, url, extracted_text, is_new FROM {sanitized_table_name} WHERE extracted_text IS NOT NULL AND (category IN ('API Error', 'Processing Error', 'Parse Error') OR summary IS NULL)"
        cur.execute(sql)
        rows = cur.fetchall()
        logging.info(f"Found {len(rows)} records to retry for LLM processing.")
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logging.error(f"Could not fetch records for LLM retry from table {sanitized_table_name}: {e}")
        return []

# --- NEW FUNCTION for --retry_llm feature ---
def update_llm_results(conn: sqlite3.Connection, table_name: str, record_id: int, category: str, summary: str):
    """Updates a record with new category and summary from a successful LLM call."""
    sanitized_table_name = _sanitize_table_name(table_name)
    sql = f"UPDATE {sanitized_table_name} SET category = ?, summary = ? WHERE id = ?"
    cur = conn.cursor()
    try:
        cur.execute(sql, (category, summary, record_id))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Failed to update record ID {record_id} with LLM results: {e}")

