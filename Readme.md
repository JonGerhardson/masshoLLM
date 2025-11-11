# masshoLLM

v0.2

A Python application to track updates on mass.gov and generate daily briefings using LLMs.

AI generated readme below. Quickstart: ```git clone```, ```cd masshoLLM```, ```pip install requirements.txt```, ```python main.py``` 

New functionality to exclude from sending to final briefing LLM pages that are not dated but the page content indicates it is old, some rudimentary fact checking steps, and more. . . Still p. sketchy, caveat emptor. 

```

  Added
   - Exclusion feature: Added an excluded column to the database schema to allow marking
     pages for exclusion from LLM processing
   - Automatic exclusion marking: LLM now flags content that references information more
     than 2 months old as "Outdated Content"
   - Database migration: Automatic schema migration to add the excluded column to existing
     tables
   - Exclusion filtering: Updated all database fetch functions to exclude records marked as
     excluded by default
   - Manual exclusion capability: Users can manually mark pages as excluded using SQLite
     browser with SQL commands like UPDATE massgov_YYYY_MM_DD SET excluded = 'yes' WHERE 
     url = 'specific_url'

  Changed
   - LLM Prompt Enhancement: Updated LLM prompt in llm_handler.py to include instructions
     for identifying content older than 2 months
   - Database schema: Added excluded TEXT column to the daily table schema
   - Database operations: Modified insert_record, update_scraped_record, and other
     database functions to handle the new excluded column
   - Fetch functions: Updated fetch_new_records, fetch_new_and_maybe_records, and related
     functions to filter out excluded records
   - LLM processing: Modified get_batch_summaries in llm_handler.py to accept database
     connection parameters and mark outdated content in DB
   - Main application flow: Updated main.py to pass database connection and table name to
     LLM processing functions

  Integration
   - Main processing flow: LLM processing in main.py now automatically flags and marks
     outdated content for future exclusion
   - Report generation: report_extras and other reporting functions now automatically
     filter out excluded pages
   - Retry functionality: LLM retry operations also respect the exclusion flag

  Files Modified
   - database.py - Added exclusion column support and migration functionality
   - llm_handler.py - Updated prompt and processing logic for outdated content
   - main.py - Updated to pass database parameters to LLM processing
   - dual_model_processor.py - Added similar functionality for consistency (though not
     currently used in main flow)
   - report_extras.py - Updated to use exclusion-filtered database queries
   - README.md - Updated documentation to reflect new exclusion feature

  Usage
   - Automatic exclusion: Content identified as more than 2 months old is automatically
     marked for exclusion
   - Manual exclusion: Use SQLite commands like UPDATE table_name SET excluded = 'yes' to
     manually exclude pages
   - Excluded pages are automatically filtered out when running report_extras or other LLM
     processing
```

## Features

- **Web Scraping**: Automatically scans mass.gov for new content
- **Database Storage**: Stores scraped data in SQLite database
- **LLM Classification**: Uses LLMs to categorize content as "new", "maybe", or "not new"
- **Briefing Generation**: Creates daily briefing reports
- **Exclusion Feature**: Ability to exclude specific pages from LLM processing

## Installation

1. Make sure you have Python 3.8+ installed
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your API keys in `.env` file (see `.env.example`)

## Usage

### Running the Scraper

The LLM got this section wrong lol. ```python main.py``` and then ```python report_extras/report_extras.py``` is hypothetically the whole process if nothing goes wrong. Things usually go wrong so see below or run ```main.py -h```

```

usage: main.py [-h] [--date DATE] [--test] [--retry] [--retry_llm] [--news-only]

A command-line agent to scrape, analyze, and report on updates from mass.gov.

options:
  -h, --help   show this help message and exit

Primary Operations:
  --date DATE  Specify a date to process in YYYY-MM-DD format. Defaults to yesterday.

Special Modes (use one at a time with --date):
  --test       Run in test mode: process first 25 URLs and use testing.db.
  --retry      Run in scraping retry mode for URLs that are missing data in the 'extracted_text' column.
  --retry_llm  Re-run LLM analysis on records that previously failed (e.g., due to API errors).
  --news-only  Only fetch data from the /news API to update a day's run, skipping the sitemap.

Examples of use:
  - Run for yesterday's date:
    python main.py

  - Run for a specific date:
    python main.py --date 2025-10-14

  - Run a quick test for a specific date's sitemap:
    python main.py --date 2025-10-14 --test

  - **NEW: Retry scraping for missing content:**
    python main.py --date 2025-10-14 --retry

  - Re-run LLM analysis on records that previously failed:
    python main.py --date 2025-10-14 --retry-llm

  - Update an existing run for a specific date with only the latest news API data:
    python main.py --date 2025-10-14 --news-only

```
### Report Extras

```bash
python report_extras.py <date>
```

This will:
- Generate a news briefing from a daily report
- Include only pages marked as 'new' by default
- Optionally include 'maybe' pages with `--include-maybe` flag

## Exclusion Feature

To exclude specific pages from being sent to the LLM during report generation:

1. After running the scraper, open your SQLite database file using your preferred DB browser
2. Navigate to the table for the specific date (e.g., `massgov_YYYY_MM_DD`)
3. Manually update the `excluded` column for pages you want to skip:

```sql
-- To exclude a specific page:
UPDATE massgov_2025_10_10 SET excluded = 'yes' WHERE url = 'https://www.mass.gov/specific-page-to-exclude';

-- To exclude multiple pages:
UPDATE massgov_2025_10_10 SET excluded = 'yes' WHERE url LIKE '%pattern-to-match%';

-- To remove exclusion for a page:
UPDATE massgov_2025_10_10 SET excluded = NULL WHERE url = 'https://www.mass.gov/specific-page';
```

The `excluded` column accepts:
- `'yes'` - Page will be excluded from LLM processing
- `NULL` or any other value - Page will be processed normally

When you run `report_extras` or other LLM processing functions, pages marked with `excluded = 'yes'` will be automatically filtered out.

## Configuration

The application uses `config.yaml` for configuration. Key settings include:

- Database file location
- LLM provider settings
- Scraping parameters
- Retry settings

## Command Line Arguments

### main.py

- `<date>`: Date in YYYY-MM-DD format
- `--scrape`: Run the scraping process
- `--llm`: Run LLM processing
- `--briefing`: Generate briefing
- `--retry`: Retry failed operations
- `--retry-llm`: Retry LLM processing for failed items

### report_extras.py

- `<date>`: Date in YYYY-MM-DD format
- `--output-dir` or `-o`: Output directory for generated files
- `--include-maybe`: Include pages marked as 'maybe' in LLM processing
- `--lmstudio`: Use LM Studio configuration for API calls

## Database Schema

The application creates daily tables with the following schema:

- `id`: Primary key
- `url`: Page URL (unique)
- `lastmodified`: Last modification date
- `filetype`: File type (html, pdf, etc.)
- `page_date`: Page publication date
- `is_new`: Classification ('yes', 'maybe', 'no')
- `category`: Content category
- `summary`: LLM-generated summary
- `extracted_text`: Raw text extracted from the page
- `source_text_hash`: SHA-256 hash of the content
- `excluded`: Exclusion flag ('yes' or NULL) - *New*

## Logs

The application generates logs in the `logs` directory:
- Daily logs for each operation
- LLM-specific logs
- Retry logs

## Troubleshooting

- Check the logs directory if you encounter issues
- Verify your API keys and network connectivity
- Ensure the database file has proper write permissions

## License

 "Commons Clause" License Condition v1.0 + Apache

 Commercial use here explicitly includes running this in a newsroom to find story leads, thanks for understanding. Email jon.gerhardson@proton.me 
