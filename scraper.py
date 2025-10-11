import requests
import logging
import dateparser
import pypdf
import io
import docx
import pandas as pd
from bs4 import BeautifulSoup
from trafilatura import extract
from typing import Optional
from datetime import datetime
import cloudscraper
from fake_useragent import UserAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
ua = UserAgent()

def create_session_with_retries() -> requests.Session:
    """Creates a cloudscraper session object."""
    return cloudscraper.create_scraper()

def fetch_page(session: requests.Session, url: str) -> Optional[str]:
    """Fetches the HTML content of a given URL."""
    try:
        headers = {'User-Agent': ua.random}
        response = session.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return None

def extract_download_page_date(html: str) -> Optional[datetime]:
    """Parses the HTML of a /doc/ landing page to find the 'Last updated' date."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        last_updated_th = soup.find('th', scope="row", string=lambda t: t and "last updated" in t.lower())
        if last_updated_th:
            next_td = last_updated_th.find_next_sibling('td')
            if next_td:
                date_span = next_td.find('span', class_='ma__listing-table__data-item')
                if date_span:
                    date_str = date_span.get_text(strip=True)
                    return dateparser.parse(date_str)
        
        last_updated_dt = soup.find('dt', string=lambda t: t and "last updated on" in t.lower())
        if last_updated_dt:
            last_updated_dd = last_updated_dt.find_next_sibling('dd')
            if last_updated_dd:
                date_str = last_updated_dd.get_text(strip=True)
                return dateparser.parse(date_str)
        return None
    except Exception as e:
        logging.error(f"Error parsing date from download page: {e}")
        return None

def find_best_date_on_page(html: str) -> Optional[datetime]:
    """Tries to find any plausible date from a standard HTML page."""
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
            date_text = date_span.get_text(strip=True).replace("Published on ", "")
            return dateparser.parse(date_text)
            
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

def download_and_extract_document_text(session: requests.Session, url: str) -> Optional[str]:
    """
    Downloads a file from a URL and extracts text.
    Handles PDF, DOCX, XLSX, XLS, and CSV file types by inspecting the file's "magic numbers".
    """
    try:
        headers = {'User-Agent': ua.random}
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        content_bytes = response.content
        file_stream = io.BytesIO(content_bytes)

        # Check for different file types using their unique starting bytes ("magic numbers")
        if content_bytes.startswith(b'%PDF'):
            try:
                pdf_reader = pypdf.PdfReader(file_stream)
                text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                if text.strip():
                    logging.info(f"Successfully extracted text from PDF: {url}")
                    return text.strip()
            except Exception as e:
                 logging.error(f"File appeared to be a PDF but failed to parse: {e}")
                 return None

        elif content_bytes.startswith(b'PK\x03\x04'):  # ZIP archive (DOCX or XLSX)
            logging.info(f"File at {url} is a ZIP-based format (DOCX or XLSX).")
            try:
                document = docx.Document(file_stream)
                text = "\n".join(para.text for para in document.paragraphs)
                if text.strip():
                    logging.info(f"Successfully extracted text from DOCX: {url}")
                    return text.strip()
            except Exception:
                logging.warning(f"Could not read as DOCX. Attempting to read as XLSX.")
                file_stream.seek(0)
                try:
                    df = pd.read_excel(file_stream, engine='openpyxl')
                    text = df.to_string()
                    if text.strip():
                        logging.info(f"Successfully extracted text from XLSX: {url}")
                        return text.strip()
                except Exception as xlsx_error:
                    logging.error(f"File was ZIP-based but failed as DOCX and XLSX: {xlsx_error}")
                    return None
        
        elif content_bytes.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'): # Legacy XLS file
            logging.info(f"File at {url} identified as a legacy XLS file.")
            try:
                df = pd.read_excel(file_stream, engine='xlrd')
                text = df.to_string()
                if text.strip():
                    logging.info(f"Successfully extracted text from XLS: {url}")
                    return text.strip()
            except Exception as xls_error:
                logging.error(f"Could not read file as XLS: {xls_error}")
                return None

        else:  # Fallback: Assume a text-based format like CSV
            logging.info(f"File at {url} not PDF/ZIP/XLS. Attempting to read as CSV.")
            try:
                try:
                    csv_text = content_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    csv_text = content_bytes.decode('latin-1')
                
                df = pd.read_csv(io.StringIO(csv_text))
                text = df.to_string()
                if text.strip():
                    logging.info(f"Successfully extracted text from CSV: {url}")
                    return text.strip()
            except Exception as csv_error:
                logging.error(f"Could not read file as CSV: {csv_error}")
                return None

        logging.warning(f"Could not extract any text from the document at {url}")
        return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download file from {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during file processing for {url}: {e}")
        return None


# --- Self-Contained Testing Block ---
if __name__ == '__main__':
    session = create_session_with_retries()
    
    # Test XLS
    test_xls = "https://www.mass.gov/doc/abcc-active-retail-licenses/download"
    print(f"\n--- Testing Document File Download (XLS): {test_xls} ---")
    xls_text = download_and_extract_document_text(session, test_xls)
    if xls_text:
        print(f"✅ SUCCESS: Extracted text (length: {len(xls_text)} chars).")
    else:
        print("❌ FAILURE: Could not process the document.")


