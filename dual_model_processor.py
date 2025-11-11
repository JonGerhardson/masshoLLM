"""
Dual-model processing module for the Massachusetts Government Agent.
This module implements the recommended approach of using 2.0-flash for initial processing
and 2.5-pro for final steps requiring high accuracy.
"""
import yaml
import logging
import time
import math
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# Import database module for marking outdated content
import database

# Import fact checker for accuracy verification
try:
    from fact_checker import verify_summary
except ImportError:
    def verify_summary(source_text: str, summary: str, url: str = "") -> Dict[str, Any]:
        # Dummy function if fact_checker is not available
        return {
            'accuracy_score': 1.0,
            'is_accurate': True,
            'confidence_score': 1.0,
            'issues': [],
            'suggestions': ['Fact checking not available']
        }

class DualModelProcessor:
    """
    Handles processing using a dual-model approach:
    - gemini-2.0-flash for initial bulk processing (high RPM, efficient)
    - gemini-2.5-pro for final high-accuracy processing (low RPM, high quality)
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the dual-model processor with configuration."""
        self._load_config(config_path)
        self._setup_models()
        
    def _load_config(self, config_path: str):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            logging.critical(f"Could not load config.yaml: {e}")
            raise
        
        llm_settings = config.get('llm_settings', {})
        
        # Model configuration
        self.flash_model_name = llm_settings.get('flash_model', 'gemini-2.0-flash')
        self.pro_model_name = llm_settings.get('pro_model', 'gemini-2.5-pro')
        
        # Rate limits (RPM)
        self.flash_rpm = llm_settings.get('flash_rpm', 15)  # gemini-2.0-flash: 15 RPM
        self.pro_rpm = llm_settings.get('pro_rpm', 2)       # gemini-2.5-pro: 2 RPM
        
        # Calculate delays
        self.flash_delay = 60.0 / self.flash_rpm
        self.pro_delay = 60.0 / self.pro_rpm
        
        # Settings
        self.use_pro_for_refinement = llm_settings.get('use_pro_for_refinement', False)
        self.truncation_length = llm_settings.get('truncation_length', 15000)
        self.max_retries = llm_settings.get('max_retries', 3)
        self.initial_backoff = llm_settings.get('initial_backoff_sec', 60)
        
        # Track last call times for rate limiting
        self.last_flash_call = 0
        self.last_pro_call = 0

    def _setup_models(self):
        """Initialize the Gemini models."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        genai.configure(api_key=api_key)
        self.flash_model = genai.GenerativeModel(self.flash_model_name)
        self.pro_model = genai.GenerativeModel(self.pro_model_name)
        
        logging.info(f"Initialized Flash model: {self.flash_model_name}")
        logging.info(f"Initialized Pro model: {self.pro_model_name}")

    def _enforce_rate_limit(self, is_pro_model: bool):
        """Enforce rate limiting based on model type."""
        current_time = time.time()
        
        if is_pro_model:
            time_elapsed = current_time - self.last_pro_call
            if time_elapsed < self.pro_delay:
                time.sleep(self.pro_delay - time_elapsed)
            self.last_pro_call = time.time()
        else:
            time_elapsed = current_time - self.last_flash_call
            if time_elapsed < self.flash_delay:
                time.sleep(self.flash_delay - time_elapsed)
            self.last_flash_call = time.time()

    def _call_model_with_retry(self, model: genai.GenerativeModel, prompt: str, is_pro_model: bool) -> Optional[str]:
        """Call the model with rate limiting and retry logic."""
        self._enforce_rate_limit(is_pro_model)
        
        for attempt in range(self.max_retries):
            try:
                logging.debug(f"Sending prompt to {'Pro' if is_pro_model else 'Flash'} model (attempt {attempt + 1})")
                response = model.generate_content(prompt)
                return response.text.strip()
            
            except google_exceptions.ResourceExhausted as e:
                if attempt < self.max_retries - 1:
                    delay = self.initial_backoff * (2 ** attempt)
                    logging.warning(f"Rate limit exceeded. Waiting {delay}s before retry {attempt + 2}...")
                    time.sleep(delay)
                else:
                    logging.error(f"Rate limit exceeded after {self.max_retries} attempts", exc_info=True)
                    return None
            
            except Exception as e:
                logging.error(f"Error calling {'Pro' if is_pro_model else 'Flash'} model: {e}", exc_info=True)
                return None
        
        return None

    def calculate_optimal_batch_size(self, model_name: str) -> tuple:
        """
        Calculate optimal batch size based on model-specific rate limits and TPM
        """
        # Model-specific limits (as per your requirements)
        model_limits = {
            'gemini-2.0-flash': {'rpm': 15, 'tpm': 1000000},  # 1M tokens per minute
            'gemini-2.5-pro': {'rpm': 2, 'tpm': 125000},      # 125K tokens per minute
            'gemini-2.5-flash': {'rpm': 10, 'tpm': 250000}    # 250K tokens per minute
        }
        
        limits = model_limits.get(model_name, model_limits['gemini-2.0-flash'])
        
        if limits['rpm'] <= 2:  # For gemini-2.5-pro with only 2 RPM
            max_chars_per_batch = min(60000, limits['tpm'] // 3)  # Use 1/3 of TPM budget per batch
            max_items_per_batch = 1  # Very small batch count due to low RPM
        elif limits['rpm'] <= 10:  # For gemini-2.5-flash with 10 RPM
            max_chars_per_batch = min(200000, limits['tpm'] // 4)  # Use 1/4 of TPM budget per batch
            max_items_per_batch = 3
        else:  # For gemini-2.0-flash with 15 RPM
            max_chars_per_batch = min(800000, limits['tpm'] // 8)  # Use 1/8 of TPM budget per batch
            max_items_per_batch = 5
        
        return max_chars_per_batch, max_items_per_batch

    def build_content_aware_batches(self, items: List[Dict[str, Any]], model_name: str) -> List[List[Dict[str, Any]]]:
        """
        Build batches based on content size rather than fixed count
        """
        max_chars_per_batch, max_items_per_batch = self.calculate_optimal_batch_size(model_name)
        
        if not items:
            return []
        
        # Sort items by content length (descending) to better pack batches
        sorted_items = sorted(items, key=lambda x: len(x.get('extracted_text', '')), reverse=True)
        
        batches = []
        current_batch = []
        current_batch_size = 0
        
        for item in sorted_items:
            item_content = item.get('extracted_text', '') or ''
            item_size = len(item_content)
            
            # Check if adding this item would exceed limits
            would_exceed_chars = current_batch_size + item_size > max_chars_per_batch
            would_exceed_count = len(current_batch) >= max_items_per_batch
            
            if would_exceed_chars or would_exceed_count:
                # Finalize current batch if not empty
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_size = 0
                
                # If single item is too large, handle it specially
                if item_size > max_chars_per_batch:
                    # For now, put large content in its own batch
                    batches.append([item])
                else:
                    # Start new batch with this item
                    current_batch.append(item)
                    current_batch_size += item_size
            else:
                # Add to current batch
                current_batch.append(item)
                current_batch_size += item_size
        
        # Add final batch if it contains items
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def classify_and_summarize_with_dual_model(self, records: List[Dict[str, Any]], db_connection=None, table_name: str = None) -> List[Dict[str, Any]]:
        """
        Use Flash model for initial classification/summarization, Pro model for refinement if needed.
        """
        if not records:
            return []
        
        logging.info(f"Starting dual-model processing for {len(records)} records...")
        logging.info(f"Using {self.flash_model_name} for initial processing and {self.pro_model_name} for final steps.")
        
        # Use content-based batching for efficient processing
        content_batches = self.build_content_aware_batches(records, self.flash_model_name)
        logging.info(f"Split {len(records)} records into {len(content_batches)} content-aware batches for initial processing.")
        
        all_results = []
        
        # Step 1: Initial processing with Flash model
        for batch_idx, batch in enumerate(content_batches):
            logging.info(f"Processing batch {batch_idx + 1}/{len(content_batches)} with Flash model ({len(batch)} items)...")
            
            batch_results = self._process_batch_with_flash(batch)
            all_results.extend(batch_results)
        
        # Step 2: Optional refinement with Pro model if configured
        if self.use_pro_for_refinement and all_results:
            logging.info(f"Refining {len(all_results)} results with {self.pro_model_name}...")
            refined_results = self._refine_with_pro_model(all_results)
            
            # Mark outdated content in DB if database connection is provided
            if db_connection and table_name:
                self.mark_outdated_content_in_db(db_connection, table_name, refined_results)
            
            return refined_results

        # Mark outdated content in DB if database connection is provided
        if db_connection and table_name:
            self.mark_outdated_content_in_db(db_connection, table_name, all_results)

        return all_results

    def _process_batch_with_flash(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch with the Flash model for initial classification and summary."""
        batch_results = []
        
        for record in batch:
            url = record['url']
            extracted_text = record.get('extracted_text', '') or ''
            
            # Truncate if necessary
            if len(extracted_text) > self.truncation_length:
                extracted_text = (
                    f"[NOTE TO EDITOR: The following document was excerpted due to extreme length.]\n\n"
                    f"{extracted_text[:self.truncation_length]}"
                )
            
            # Create prompt for classification and summary
            prompt = f"""
            You are an automated content classifier for Massachusetts government updates.
            Today's date is {datetime.now().strftime('%Y-%m-%d')}.
            
            Please classify and summarize the following content:
            
            URL: {url}
            CONTENT: {extracted_text}
            
            CRITICAL: Before categorizing, carefully check if this content references information more than TWO MONTHS old. 
            If the content is clearly outdated (older than 2 months), mark it as "Outdated Content" in the category field.

            Respond with a single, valid JSON object:
            {{
              "url": "{url}",
              "category": "One of: Meeting Announcement, Meeting Materials, New Announcement, Press Release, Recent Update, Timeless Info, New Document, Outdated Content",
              "summary": "A concise, neutral summary of the content in 1-2 sentences",
              "justification": "Brief reason for this classification"
            }}
            """
            
            response = self._call_model_with_retry(self.flash_model, prompt, is_pro_model=False)
            
            if response:
                try:
                    result = json.loads(response)
                    
                    # Perform fact checking on the generated summary
                    verification_result = verify_summary(extracted_text, result.get('summary', ''), url)
                    
                    if not verification_result.get('is_accurate', True):
                        # Add detailed warning to summary if fact check fails
                        detailed_issues = verification_result.get('detailed_issues', [])
                        if detailed_issues:
                            issues_text = "; ".join(detailed_issues[:2])  # Limit to first 2 issues to avoid clutter
                            if len(detailed_issues) > 2:
                                issues_text += f"; +{len(detailed_issues) - 2} more issues"
                            accuracy_warning = f"\n[FACT CHECK WARNING: {len(detailed_issues)} accuracy issues - {issues_text}. {verification_result.get('suggestions', ['Review for accuracy'])[0]}]"
                        else:
                            accuracy_warning = f"\n[FACT CHECK WARNING: {len(verification_result.get('issues', []))} accuracy issues detected. {verification_result.get('suggestions', ['Review for accuracy'])[0]}]"
                        result['summary'] += accuracy_warning
                        logging.warning(f"Fact check failed for {url}: {detailed_issues if detailed_issues else verification_result.get('issues', [])}")
                    
                    batch_results.append(result)
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse JSON response for {url}")
                    # Fallback: return basic info
                    batch_results.append({
                        'url': url,
                        'category': 'Parsing Error',
                        'summary': 'Content failed to parse',
                        'justification': 'JSON parsing failed'
                    })
            else:
                logging.error(f"Failed to get response for {url}")
                batch_results.append({
                    'url': url,
                    'category': 'API Error',
                    'summary': 'Failed to process content',
                    'justification': 'API call failed'
                })
        
        return batch_results

    def _refine_with_pro_model(self, initial_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use Pro model to refine classifications and summaries if higher accuracy is needed."""
        refined_results = []
        
        # Group results for batch processing to minimize Pro model calls
        for result in initial_results:
            url = result['url']
            original_category = result['category']
            original_summary = result['summary']
            
            # Create refinement prompt
            refinement_prompt = f"""
            You are a senior editor reviewing an automatically generated classification and summary.
            Original classification: {original_category}
            Original summary: {original_summary}
            
            Please provide a more accurate classification and summary for this content:
            {result.get('url', 'URL not provided')}
            
            Return a single JSON object:
            {{
              "url": "{url}",
              "category": "Refined category",
              "summary": "More accurate summary",
              "justification": "Reason for this classification"
            }}
            """
            
            response = self._call_model_with_retry(self.pro_model, refinement_prompt, is_pro_model=True)
            
            if response:
                try:
                    refined_result = json.loads(response)
                    refined_results.append(refined_result)
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse refinement response for {url}")
                    refined_results.append(result)  # Keep original if refinement fails
            else:
                logging.error(f"Failed to refine result for {url}")
                refined_results.append(result)  # Keep original if refinement fails
        
        return refined_results

    def mark_outdated_content_in_db(self, db_connection, table_name: str, records: List[Dict[str, Any]]):
        """Mark records with 'Outdated Content' category as excluded in the database."""
        cursor = db_connection.cursor()
        
        for record in records:
            if record.get('category') == 'Outdated Content':
                url = record['url']
                try:
                    # Update the record to mark it as excluded
                    update_sql = f"UPDATE {table_name} SET excluded = 'yes' WHERE url = ?"
                    cursor.execute(update_sql, (url,))
                    logging.info(f"Marked {url} as excluded due to outdated content")
                except Exception as e:
                    logging.error(f"Failed to mark {url} as excluded: {e}")
        
        db_connection.commit()
        logging.info(f"Updated database to mark {len([r for r in records if r.get('category') == 'Outdated Content'])} records as excluded")

    def generate_final_briefing(self, processed_data: List[Dict[str, Any]], target_date: str) -> Optional[str]:
        """Use Pro model to generate the final briefing with high accuracy."""
        logging.info(f"Generating final briefing for {target_date} using {self.pro_model_name}...")
        
        if not processed_data:
            logging.warning("No processed data to generate briefing from.")
            return None
        
        # Prepare the data for briefing generation
        news_items = [item for item in processed_data if item.get('category') != 'Meeting Announcement']
        meeting_items = [item for item in processed_data if item.get('category') == 'Meeting Announcement']
        
        # Create briefing prompt
        briefing_prompt = f"""
        You are a senior news editor creating "The Massachusetts Logfiler" briefing for {target_date}.
        
        Here are the processed items to include in the briefing:
        
        NEWS ITEMS:
        {json.dumps(news_items, indent=2)}
        
        MEETING ANNOUNCEMENTS:
        {json.dumps(meeting_items, indent=2)}
        
        Please create a professional briefing document with the following sections:
        1. Top stories (select most important items)
        2. Upcoming meetings (for meeting items)
        3. Other updates by category
        
        Format as markdown with clear headings and include source URLs.
        """
        
        response = self._call_model_with_retry(self.pro_model, briefing_prompt, is_pro_model=True)
        return response