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
        
        # Automatically migrate existing tables to include the 'excluded' column
        migrate_existing_tables(conn)
        
        return conn
    except sqlite3.Error as e:
        logging.error(f"Error connecting to database: {e}")
    return None

def create_daily_table(conn: sqlite3.Connection, table_name: str):
    """Creates a new table for the daily run if it doesn't exist."""
    sanitized_table_name = _sanitize_table_name(table_name)
    # Schema now includes the 'extracted_text' column for the retry feature.
    # Added source_text_hash to track exact source content for verification
    # Added 'excluded' column for marking pages to be excluded from report_extras
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
        extracted_text TEXT,
        source_text_hash TEXT,  -- For tracking original content and detecting changes
        excluded TEXT          -- For marking pages to be excluded from report_extras ('yes' or NULL/empty)
    );
    """
    try:
        c = conn.cursor()
        c.execute(sql_create_table)
        # Ensure the table has the latest schema in case it was already created without the new column
        add_excluded_column_if_missing(conn, table_name)
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

import hashlib

def calculate_content_hash(content: str) -> str:
    """Calculate SHA-256 hash of content for verification."""
    if content:
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    return None

def insert_record(conn: sqlite3.Connection, table_name: str, record_data: Dict[str, Any]):
    """Inserts a single processed record into the database."""
    sanitized_table_name = _sanitize_table_name(table_name)
    # Updated SQL statement to include the new 'extracted_text' and 'source_text_hash' columns.
    sql = f''' INSERT INTO {sanitized_table_name}(url, lastmodified, filetype, page_date, is_new, category, summary, extracted_text, source_text_hash, excluded)
               VALUES(?,?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    try:
        extracted_text = record_data.get('extracted_text')
        source_text_hash = calculate_content_hash(extracted_text) if extracted_text else None
        
        cur.execute(sql, (
            record_data['url'],
            record_data['lastmodified'],
            record_data['filetype'],
            record_data['page_date'],
            record_data['is_new'],
            record_data.get('category'),
            record_data['summary'],
            extracted_text,
            source_text_hash,
            record_data.get('excluded')  # Default to None/NULL if not provided
        ))
        conn.commit()
    except sqlite3.IntegrityError:
        logging.warning(f"URL already exists in database, skipping insert: {record_data['url']}")
    except sqlite3.Error as e:
        logging.error(f"Error inserting record for URL {record_data['url']}: {e}")

def fetch_new_and_maybe_records(conn: sqlite3.Connection, table_name: str) -> List[Dict[str, Any]]:
    """Fetches all records marked as 'yes' or 'maybe' for reporting, excluding those marked as excluded."""
    sanitized_table_name = _sanitize_table_name(table_name)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        # Include extracted_text for fact-checking purposes
        # Exclude records where excluded column is 'yes'
        cur.execute(f"SELECT * FROM {sanitized_table_name} WHERE is_new IN ('yes', 'maybe') AND (excluded IS NULL OR excluded != 'yes')")
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logging.error(f"Could not fetch records for report from table {sanitized_table_name}: {e}")
        return []

def fetch_new_records(conn: sqlite3.Connection, table_name: str) -> List[Dict[str, Any]]:
    """Fetches all records marked as 'yes' for reporting."""
    sanitized_table_name = _sanitize_table_name(table_name)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        # Include extracted_text for fact-checking purposes
        cur.execute(f"SELECT * FROM {sanitized_table_name} WHERE is_new = 'yes'")
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logging.error(f"Could not fetch 'new' records for report from table {sanitized_table_name}: {e}")
        return []

def fetch_new_records(conn: sqlite3.Connection, table_name: str) -> List[Dict[str, Any]]:
    """Fetches all records marked as 'yes' for reporting, excluding those marked as excluded."""
    sanitized_table_name = _sanitize_table_name(table_name)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        # Exclude records where excluded column is 'yes'
        cur.execute(f"SELECT * FROM {sanitized_table_name} WHERE is_new = 'yes' AND (excluded IS NULL OR excluded != 'yes')")
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logging.error(f"Could not fetch 'new' records for report from table {sanitized_table_name}: {e}")
        return []

# --- NEW FUNCTION for --retry feature (Scraping) ---
def fetch_urls_for_scraping_retry(conn: sqlite3.Connection, table_name: str) -> List[Dict[str, str]]:
    """
    Fetches records from the table where 'extracted_text' is NULL or an empty string.
    Returns URL and lastmodified for re-processing.
    """
    sanitized_table_name = _sanitize_table_name(table_name)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        # Check for NULL OR an empty string ''
        sql = f"""
        SELECT url, lastmodified
        FROM {sanitized_table_name} 
        WHERE extracted_text IS NULL OR extracted_text = ''
        """
        cur.execute(sql)
        rows = cur.fetchall()
        logging.info(f"Found {len(rows)} records with no extracted text for scraping retry.")
        # Only return URL and lastmodified, as these are needed for process_url
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logging.error(f"Could not fetch records for scraping retry from table {sanitized_table_name}: {e}")
        return []

# --- NEW FUNCTION for --retry feature (Updating database) ---
def update_scraped_record(conn: sqlite3.Connection, table_name: str, record_data: Dict[str, Any]):
    """
    Updates an existing record with newly scraped/extracted data 
    (filetype, page_date, is_new, extracted_text, source_text_hash).
    """
    sanitized_table_name = _sanitize_table_name(table_name)
    sql = f"""
    UPDATE {sanitized_table_name} 
    SET 
        lastmodified = ?, 
        filetype = ?, 
        page_date = ?, 
        is_new = ?, 
        extracted_text = ?, 
        source_text_hash = ?,
        excluded = ?
    WHERE url = ?
    """
    cur = conn.cursor()
    try:
        extracted_text = record_data.get('extracted_text')
        source_text_hash = calculate_content_hash(extracted_text) if extracted_text else None
        
        cur.execute(sql, (
            record_data['lastmodified'],
            record_data['filetype'],
            record_data['page_date'],
            record_data['is_new'],
            extracted_text,
            source_text_hash,
            record_data.get('excluded'),  # Default to None/NULL if not provided
            record_data['url']
        ))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Failed to update scraped record for URL {record_data['url']}: {e}")

# --- Existing LLM retry functions are kept below ---
def fetch_records_for_llm_retry(conn: sqlite3.Connection, table_name: str) -> List[Dict[str, Any]]:
    """Fetches records that have stored text but failed LLM processing."""
    sanitized_table_name = _sanitize_table_name(table_name)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        # Fetches records where the text was extracted but the LLM failed (API Error) or wasn't run.
        # Also exclude records marked as excluded
        sql = f"SELECT id, url, extracted_text, is_new FROM {sanitized_table_name} WHERE extracted_text IS NOT NULL AND (category IN ('API Error', 'Processing Error', 'Parse Error') OR summary IS NULL) AND (excluded IS NULL OR excluded != 'yes')"
        cur.execute(sql)
        rows = cur.fetchall()
        logging.info(f"Found {len(rows)} records to retry for LLM processing.")
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logging.error(f"Could not fetch records for LLM retry from table {sanitized_table_name}: {e}")
        return []

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

def add_excluded_column_if_missing(conn: sqlite3.Connection, table_name: str):
    """Add the excluded column to an existing table if it doesn't exist."""
    sanitized_table_name = _sanitize_table_name(table_name)
    
    # Check if the 'excluded' column already exists
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({sanitized_table_name})")
    columns = [column[1] for column in cur.fetchall()]
    
    if 'excluded' not in columns:
        try:
            # Add the excluded column to the table
            alter_sql = f"ALTER TABLE {sanitized_table_name} ADD COLUMN excluded TEXT"
            cur.execute(alter_sql)
            conn.commit()
            logging.info(f"Added 'excluded' column to table '{sanitized_table_name}'")
        except sqlite3.Error as e:
            logging.error(f"Error adding 'excluded' column to table '{sanitized_table_name}': {e}")
            return False
    else:
        logging.info(f"'excluded' column already exists in table '{sanitized_table_name}'")
    
    return True

def ensure_table_schema(conn: sqlite3.Connection, table_name: str):
    """Ensure the table has the latest schema with all required columns."""
    add_excluded_column_if_missing(conn, table_name)

def migrate_existing_tables(conn: sqlite3.Connection):
    """Migrate all existing tables in the database to include the excluded column."""
    cur = conn.cursor()
    
    # Get all table names that follow the massgov pattern
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'massgov_%'")
    tables = cur.fetchall()
    
    migrated_count = 0
    for table in tables:
        table_name = table[0]
        if add_excluded_column_if_missing(conn, table_name):
            migrated_count += 1
    
    logging.info(f"Migration completed: {migrated_count} tables checked/updated for the 'excluded' column")


def fetch_new_and_maybe_records_with_exclusions(conn: sqlite3.Connection, table_name: str) -> List[Dict[str, Any]]:
    """Fetches all records marked as 'yes' or 'maybe' for reporting, excluding those marked as excluded."""
    sanitized_table_name = _sanitize_table_name(table_name)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        # Include extracted_text for fact-checking purposes
        # Exclude records where excluded column is 'yes'
        cur.execute(f"SELECT * FROM {sanitized_table_name} WHERE is_new IN ('yes', 'maybe') AND (excluded IS NULL OR excluded != 'yes')")
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logging.error(f"Could not fetch records for report from table {sanitized_table_name}: {e}")
        return []

def fetch_new_records_with_exclusions(conn: sqlite3.Connection, table_name: str) -> List[Dict[str, Any]]:
    """Fetches all records marked as 'yes' for reporting, excluding those marked as excluded."""
    sanitized_table_name = _sanitize_table_name(table_name)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        # Exclude records where excluded column is 'yes'
        cur.execute(f"SELECT * FROM {sanitized_table_name} WHERE is_new = 'yes' AND (excluded IS NULL OR excluded != 'yes')")
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logging.error(f"Could not fetch 'new' records for report from table {sanitized_table_name}: {e}")
        return []

