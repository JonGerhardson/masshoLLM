import logging
import argparse
import yaml
import os
import re
import time
import google.generativeai as genai
from typing import Optional, List, Dict

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    """Loads the main configuration file."""
    try:
        with open("config.yaml", 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logging.critical(f"CRITICAL ERROR reading config.yaml: {e}")
        return None

def parse_urls_by_section(report_content: str) -> Dict[str, List[str]]:
    """
    Parses a markdown report to extract URLs, organizing them by their section.
    """
    urls_by_section = {}
    
    # Define sections in the order they should be processed
    sections = [
        "## New announcements",
        "## New documents",
        "## Might be new?"
    ]

    seen_urls = set()
    full_text_for_parsing = report_content + "\n## END" # Add a final delimiter

    for i in range(len(sections)):
        section_start_marker = sections[i]
        section_end_marker = sections[i+1] if i + 1 < len(sections) else "## END"
        
        # Use a clean version of the section name as the dictionary key
        section_key = section_start_marker.strip('# ').capitalize()
        
        section_urls_list = []
        
        # Regex to capture content between two section headers
        try:
            section_content_match = re.search(
                f"{re.escape(section_start_marker)}(.*?){re.escape(section_end_marker)}",
                full_text_for_parsing,
                re.DOTALL | re.IGNORECASE
            )
            if section_content_match:
                section_content = section_content_match.group(1)
                # Find all URLs within this section
                section_urls = re.findall(r'\(https?://www.mass.gov[^\s\)]+\)', section_content)
                for url in section_urls:
                    cleaned_url = url.strip("()")
                    if cleaned_url not in seen_urls:
                        section_urls_list.append(cleaned_url)
                        seen_urls.add(cleaned_url)
            
            if section_urls_list:
                 urls_by_section[section_key] = section_urls_list

        except re.error as e:
            logging.error(f"Regex error while parsing section '{section_start_marker}': {e}")
            continue

    logging.info(f"Found and categorized URLs into {len(urls_by_section)} sections.")
    return urls_by_section

def get_briefing_part(urls_chunk: List[str], api_key: str, current_date: str) -> Optional[str]:
    """Sends a chunk of URLs to Gemini to be 'fleshed out' with date context."""
    try:
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel(model_name='gemini-2.5-pro')

        url_list_str = "\n".join(urls_chunk)

        prompt = (
            f"Today is {current_date}. You are a news writer preparing today's news brief. "
            "Your task is to access and summarize the content at the following URLs. "
            "Focus ONLY on the information presented on the linked pages. "
            "Information more than a week old from today's date is out of scope unless it is essential context for a current event. "
            "For each URL, provide a detailed, journalistic paragraph summarizing its key, timely information. "
            "Maintain a neutral tone."
            f"\n\n--- URLS TO PROCESS ---\n{url_list_str}"
        )

        logging.info(f"Sending {len(urls_chunk)} URLs to Gemini for initial analysis...")
        
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        logging.error(f"API call failed for chunk of {len(urls_chunk)} URLs: {e}", exc_info=True)
        return None

def format_final_briefing(combined_parts: str, all_urls: List[str], api_key: str, current_date: str) -> Optional[str]:
    """Sends the combined parts to Gemini for final formatting with date context."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name='gemini-2.5-pro')
        
        url_list_str = "\n".join(all_urls)

        prompt = (
            f"Today is {current_date}. You are a senior news editor. "
            "The following text contains raw notes for today's news briefing, organized by category. "
            "Your task is to synthesize, edit, and format these notes into a single, cohesive, and professional news briefing document. "
            "Ensure the final output is timely and relevant to today's date. Information more than a week old should generally be excluded unless it is critical context. "
            "Organize the content logically with clear headings. "
            "When you reference a specific story, you MUST include a markdown hyperlink to the corresponding source URL from the list provided below. "
            "Ensure a consistent, journalistic tone throughout."
            f"\n\n--- SOURCE URLS ---\n{url_list_str}"
            "\n\n--- RAW BRIEFING NOTES ---\n\n"
        )
        
        generation_config = genai.GenerationConfig(temperature=0.2)
        
        logging.info("Sending combined content to Gemini for final formatting...")
        response = model.generate_content(prompt + combined_parts, generation_config=generation_config)
        return response.text
    except Exception as e:
        logging.error(f"Final formatting API call failed: {e}", exc_info=True)
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate a news briefing from a daily report.")
    parser.add_argument("date", help="The date of the report to process, in YYYY-MM-DD format.")
    args = parser.parse_args()
    
    csv_date_str = args.date
    report_filename = f"report_{csv_date_str}.md"
    briefing_filename = f"briefing_{csv_date_str}.md"
    raw_filename = briefing_filename.replace('.md', '_full_raw.md')

    logging.info(f"--- Starting News Briefing Generation for {csv_date_str} ---")

    if not os.path.exists(report_filename):
        logging.critical(f"Report file not found: {report_filename}. Cannot generate briefing.")
        return

    config = load_config()
    if not config:
        return
        
    api_key = os.environ.get("GEMINI_API_KEY") or config.get('llm_settings', {}).get('api_keys', {}).get('gemini')
    if not api_key or "YOUR_" in api_key:
        logging.error("Gemini API key is not configured.")
        return

    with open(report_filename, 'r', encoding='utf-8') as f:
        report_content = f.read()

    urls_by_section = parse_urls_by_section(report_content)
    
    if not urls_by_section:
        logging.warning("No URLs found in the report. Cannot generate briefing.")
        return

    all_processed_urls = []
    section_contents = []
    
    section_order = ["New announcements", "New documents", "Might be new?"]

    for section_name in section_order:
        if section_name not in urls_by_section:
            continue

        section_urls = urls_by_section[section_name]
        logging.info(f"--- Processing section: {section_name} ({len(section_urls)} URLs) ---")
        
        url_chunks = [section_urls[i:i + 20] for i in range(0, len(section_urls), 20)][:3]
        
        section_briefing_parts = []
        for i, chunk in enumerate(url_chunks):
            logging.info(f"Processing chunk {i+1}/{len(url_chunks)} for section '{section_name}'...")
            # Pass the date to the function
            part = get_briefing_part(chunk, api_key, csv_date_str)
            if part:
                section_briefing_parts.append(part)
                all_processed_urls.extend(chunk)
            
            if i < len(url_chunks) - 1:
                time.sleep(2)
        
        if section_briefing_parts:
            full_section_content = "\n\n".join(section_briefing_parts)
            section_contents.append(f"## {section_name}\n\n{full_section_content}")

    if not section_contents:
        logging.error("Failed to get any briefing parts from the API for any section.")
        return

    combined_content = "\n\n---\n\n".join(section_contents)
    
    try:
        with open(raw_filename, 'w', encoding='utf-8') as f:
            f.write(combined_content)
        logging.info(f"Successfully saved structured raw briefing content to {raw_filename}")
    except IOError as e:
        logging.error(f"Failed to write raw briefing file: {e}")

    # Pass the date to the final formatting function
    final_briefing = format_final_briefing(combined_content, all_processed_urls, api_key, csv_date_str)

    if final_briefing:
        try:
            with open(briefing_filename, 'w', encoding='utf-8') as f:
                f.write(final_briefing)
            logging.info(f"Successfully saved final news briefing to {briefing_filename}")
        except IOError as e:
            logging.error(f"Failed to write briefing file: {e}")
    else:
        logging.error("Failed to generate the final news briefing.")

    logging.info("--- News Briefing Generation Complete ---")

if __name__ == "__main__":
    main()
