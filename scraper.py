import requests
import logging
import dateparser
import pypdf
import io
import pandas as pd
import xlrd # For .xls files
import json
from bs4 import BeautifulSoup
from trafilatura import extract
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

# --- New Imports for Robustness ---
import cloudscraper
from fake_useragent import UserAgent

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Custom Exception for 403 Errors ---
class Scraper403Error(Exception):
    """Custom exception for 403 Forbidden errors to handle them specifically."""
    pass

# --- Global UserAgent Instance ---
ua = UserAgent()

def create_session_with_retries() -> requests.Session:
    """Creates a cloudscraper session object with more robust headers to bypass anti-bot measures."""
    session = cloudscraper.create_scraper()
    # --- UPGRADE: Add a more comprehensive set of default headers to the session ---
    session.headers.update({
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
    })
    return session

def fetch_page(session: requests.Session, url: str) -> Optional[str]:
    """Fetches the HTML content of a given URL using a robust session object."""
    try:
        # --- UPGRADE: Add a Referer header to make the request look more natural ---
        headers = {
            'User-Agent': ua.random,
            'Referer': 'https://www.mass.gov/'
        }
        response = session.get(url, headers=headers, timeout=20)
        
        if response.status_code == 403:
            raise Scraper403Error(f"403 Forbidden error for URL: {url}")
            
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return None

def extract_download_page_date(html: str) -> Optional[datetime]:
    """Parses the HTML of a /doc/ landing page to find the 'Last updated' or 'Date' field."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Check for "last updated" pattern
        last_updated_th = soup.find('th', scope="row", string=lambda t: t and "last updated" in t.lower())
        if last_updated_th:
            next_td = last_updated_th.find_next_sibling('td')
            if next_td:
                date_span = next_td.find('span', class_='ma__listing-table__data-item')
                if date_span:
                    return dateparser.parse(date_span.get_text(strip=True))

        # --- ADDED: Check for "Date:" pattern ---
        date_th = soup.find('th', scope="row", string=lambda t: t and "date:" in t.lower())
        if date_th:
            next_td = date_th.find_next_sibling('td')
            if next_td:
                date_span = next_td.find('span', class_='ma__listing-table__data-item')
                if date_span:
                    return dateparser.parse(date_span.get_text(strip=True))
        
        # Check for "last updated on" pattern
        last_updated_dt = soup.find('dt', string=lambda t: t and "last updated on" in t.lower())
        if last_updated_dt:
            last_updated_dd = last_updated_dt.find_next_sibling('dd')
            if last_updated_dd:
                return dateparser.parse(last_updated_dd.get_text(strip=True))
        
        return None
    except Exception as e:
        logging.error(f"Error parsing date from download page: {e}")
        return None


def find_best_date_on_page(html: str) -> Optional[datetime]:
    """Tries to find any plausible date from a standard HTML page using multiple methods."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        meta_tag = soup.find('meta', property='article:published_time')
        if meta_tag and meta_tag.get('content'):
            return dateparser.parse(meta_tag['content'])
        
        press_date_div = soup.find('div', class_='ma__press-status__date')
        if press_date_div:
            return dateparser.parse(press_date_div.get_text(strip=True))
            
        date_span = soup.find('span', class_='ma-page-header__published-date')
        if date_span:
            return dateparser.parse(date_span.get_text(strip=True).replace("Published on ", ""))
            
        time_tag = soup.find('time', attrs={'datetime': True})
        if time_tag:
            return dateparser.parse(time_tag['datetime'])
        return None
    except Exception as e:
        logging.error(f"Error finding a generic date on page: {e}")
        return None

def extract_article_content(html: str) -> Optional[str]:
    """Uses 'trafilatura' to extract main article content."""
    return extract(html, include_comments=False, include_tables=False)

def download_and_extract_document_text(session: requests.Session, url: str) -> Dict[str, Optional[str]]:
    """
    Downloads a file, identifies its type based on content, and extracts all text from it.
    This function is now only called for supported file types.
    """
    try:
        # --- UPGRADE: Add a Referer header to the download request as well ---
        headers = {
            'User-Agent': ua.random,
            'Referer': 'https://www.mass.gov/'
            }
        response = session.get(url, headers=headers, timeout=30)
        
        if response.status_code == 403:
            raise Scraper403Error(f"403 Forbidden error for URL: {url}")
            
        response.raise_for_status()
        content_bytes = response.content
        file_stream = io.BytesIO(content_bytes)

        # Identify file type by magic number for accuracy
        magic_number = content_bytes[:4]
        text = None
        file_type = "Unknown"

        if magic_number == b'%PDF':
            file_type = "PDF"
            pdf_reader = pypdf.PdfReader(file_stream)
            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        elif magic_number == b'PK\x03\x04': # ZIP archive (DOCX, XLSX)
            try:
                file_type = "XLSX"
                df = pd.read_excel(file_stream, sheet_name=None, engine='openpyxl')
                text = "\n\n".join(f"Sheet: {name}\n{sheet.to_string()}" for name, sheet in df.items())
            except Exception:
                try:
                    import docx
                    file_type = "DOCX"
                    doc = docx.Document(file_stream)
                    text = "\n".join([para.text for para in doc.paragraphs])
                except Exception as e:
                    logging.warning(f"File at {url} is a ZIP archive but not a readable XLSX or DOCX: {e}")
        elif magic_number == b'\xd0\xcf\x11\xe0': # Legacy XLS file
             file_type = "XLS"
             workbook = xlrd.open_workbook(file_contents=content_bytes)
             text = ""
             for sheet in workbook.sheets():
                 text += f"Sheet: {sheet.name}\n"
                 for row_idx in range(sheet.nrows):
                     text += "\t".join(str(cell.value) for cell in sheet.row(row_idx)) + "\n"
        else: # Fallback to CSV/plain text
            try:
                file_type = "CSV"
                text = pd.read_csv(file_stream).to_string()
            except Exception as e:
                logging.error(f"Could not read file as CSV: {e}")

        return {"content_for_llm": text.strip() if text else None, "filetype": file_type}

    except pypdf.errors.PdfReadError:
        logging.warning(f"File at {url} appears to be a PDF but is unreadable.")
        return {"content_for_llm": None, "filetype": "PDF (unreadable)"}
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download file from {url}: {e}")
        return {"content_for_llm": None, "filetype": "Download Failed"}
    except Exception as e:
        logging.error(f"An unexpected error occurred during file processing for {url}: {e}", exc_info=True)
        return {"content_for_llm": None, "filetype": "Processing Error"}

def fetch_press_releases(session: requests.Session) -> List[Dict[str, Any]]:
    """Fetches the latest press releases from the Mass.gov news API."""
    url = "https://www.mass.gov/api/v1/news"
    logging.info(f"Fetching press releases from {url}")
    try:
        headers = {
            'User-Agent': ua.random,
            'Accept': 'application/json'
        }
        response = session.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return data
        else:
            logging.error("Press release API response is not in the expected list format.")
            return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching press releases from API: {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from press release API: {e}")
        return []
