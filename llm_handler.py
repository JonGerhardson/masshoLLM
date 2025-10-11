import os
import yaml
import logging
import requests
import google.generativeai as genai
import json
from datetime import datetime
from fake_useragent import UserAgent

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global UserAgent Instance ---
ua = UserAgent()

# --- Constants ---
TODAY_DATE_STR = datetime.now().strftime("%Y-%m-%d")

PROMPT_FOR_YES = f"""
Today's date is {TODAY_DATE_STR}. The following text is from a newly updated document from the mass.gov website.
What is this document about? Provide a concise, neutral, two-sentence summary suitable for a news lead.
"""

PROMPT_FOR_MAYBE = f"""
Today's date is {TODAY_DATE_STR}. The following text is from a mass.gov webpage with an unknown publication date.
Based on the text, is this likely a new announcement, a recent update, or timeless informational content?
What is the page about? Provide a concise, neutral, two-sentence summary suitable for a news lead.
"""

PROMPT_FOR_BATCH = f"""
You are a helpful assistant that summarizes documents. You will be given a list of documents, each prefixed with its unique URL.
For each document, provide a concise, neutral, two-sentence summary suitable for a news lead.
If a document's content suggests it is an old or timeless informational page rather than a new announcement, please note that in the summary.

Your entire response MUST be a single, valid JSON object.
The keys of the JSON object must be the document URLs, and the values must be the corresponding two-sentence summaries.

Example Response Format:
{{
  "https://www.mass.gov/doc/some-document/download": "This document outlines the new tax-lien case procedures effective October 2025. It details the steps for municipalities to file and manage tax liens through the updated online portal.",
  "https://www.mass.gov/news/another-article": "This press release announces new COVID-19 vaccine guidance. The guidance is evidence-based and aims to ensure access for all residents."
}}

Here are the documents to summarize:
"""

_config = None

def _load_config():
    """Loads the config.yaml file and caches it."""
    global _config
    if _config is None:
        try:
            with open("config.yaml", 'r') as f:
                _config = yaml.safe_load(f)
        except FileNotFoundError:
            logging.critical("CRITICAL ERROR: config.yaml not found. The application cannot run.")
            raise
        except yaml.YAMLError as e:
            logging.critical(f"CRITICAL ERROR: Error parsing config.yaml: {e}")
            raise
    return _config

# --- Internal LLM Call Functions ---

def _call_gemini(content, prompt, api_key, char_limit, model_name):
    """Handles the API call to Google Gemini."""
    if model_name == 'gemini-pro':
        model_name = 'gemini-2.5-flash' # Silently upgrade to the correct model

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        full_prompt = f"{prompt}\n\n---\n\nDOCUMENT CONTENT:\n{content[:char_limit]}"
        
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}")
        return None

def _call_openai_compatible(content, prompt, api_key, char_limit, endpoint, model_name=""):
    """Handles API calls to services like OpenRouter and LMStudio."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": ua.random
    }
    
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content[:char_limit]}
        ]
    }
    
    try:
        response = requests.post(endpoint, headers=headers, json=body, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        summary = data['choices'][0]['message']['content']
        return summary.strip()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling OpenAI-compatible API at {endpoint}: {e}")
        return None
    except (KeyError, IndexError) as e:
        logging.error(f"Error parsing response from API at {endpoint}: {e}")
        return None

# --- Public-Facing Functions ---

def get_summary(content, is_maybe):
    """Generates a summary for a single piece of content."""
    try:
        config = _load_config()['llm_settings']
    except (TypeError, KeyError):
        logging.error("LLM settings are missing or malformed in config.yaml.")
        return None

    provider = config.get('provider')
    api_keys = config.get('api_keys', {})
    endpoints = config.get('endpoints', {})
    models = config.get('models', {})
    char_limit = config.get('character_limit', 8000)
    prompt = PROMPT_FOR_MAYBE if is_maybe else PROMPT_FOR_YES
    
    logging.info(f"Requesting summary using provider: {provider}")

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or api_keys.get('gemini')
        model_name = models.get('gemini', 'gemini-2.5-flash')
        return _call_gemini(content, prompt, api_key, char_limit, model_name)

    elif provider == "openrouter":
        api_key = api_keys.get('openrouter')
        endpoint = endpoints.get('openrouter')
        model_name = models.get('openrouter', 'google/gemini-2.5-flash')
        return _call_openai_compatible(content, prompt, api_key, char_limit, endpoint, model_name=model_name)
        
    elif provider == "lmstudio":
        endpoint = endpoints.get('lmstudio')
        model_name = models.get('lmstudio', 'local-model')
        return _call_openai_compatible(content, prompt, "not-needed", char_limit, endpoint, model_name=model_name)

    else:
        logging.error(f"Unknown LLM provider '{provider}' specified in config.yaml.")
        return None

def get_batch_summaries(batch_requests):
    """Generates summaries for a batch of content in a single API call to Gemini."""
    try:
        config = _load_config()['llm_settings']
    except (TypeError, KeyError):
        logging.error("LLM settings are missing or malformed in config.yaml.")
        return {}

    provider = config.get('provider')
    if provider != 'gemini':
        logging.warning(f"Batch mode is only optimized for Gemini. Provider is '{provider}'. Falling back to single requests.")
        summaries = {}
        for req in batch_requests:
            summary = get_summary(req['content'], req['is_maybe'])
            if summary:
                summaries[req['url']] = summary
        return summaries

    full_prompt_content = []
    for req in batch_requests:
        full_prompt_content.append(f"\n---\n\nDOCUMENT URL: {req['url']}\n\n{req['content']}")

    batch_prompt = PROMPT_FOR_BATCH + "".join(full_prompt_content)

    api_keys = config.get('api_keys', {})
    models = config.get('models', {})
    api_key = os.environ.get("GEMINI_API_KEY") or api_keys.get('gemini')
    if not api_key or "YOUR_" in api_key:
        logging.error("Gemini API key is not configured.")
        return {}

    model_name = models.get('gemini', 'gemini-2.5-flash')
    char_limit = config.get('character_limit', 8000) * len(batch_requests)

    logging.info(f"Sending a batch of {len(batch_requests)} documents to Gemini in a single request.")
    raw_response = _call_gemini("", batch_prompt, api_key, char_limit, model_name)

    if not raw_response:
        logging.error("Received no response from Gemini for the batch request.")
        return {}

    try:
        cleaned_response = raw_response.strip().replace("```json", "").replace("```", "").strip()
        summaries = json.loads(cleaned_response)
        logging.info(f"Successfully parsed {len(summaries)} summaries from the batch response.")
        return summaries
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON response from Gemini batch request: {e}")
        logging.debug(f"Raw response was:\n{raw_response}")
        return {}


