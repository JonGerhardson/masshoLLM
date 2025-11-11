# masshoLLM

v0.2

A Python application to track updates on mass.gov and generate daily briefings using LLMs.

AI generated readme below. Quickstart: ```git clone```, ```cd masshoLLM```, ```pip install requirements.txt```, ```python main.py``` 

New functionality to exclude from sending to final briefing LLM pages that are not dated but the page content indicates it is old, some rudimentary fact checking steps, and more. . . Still p. sketchy, caveat emptor. 

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

```bash
python main.py <date> --scrape
```

The scraper will:
- Crawl mass.gov for updated documents
- Extract content from web pages
- Store results in the database

### LLM Processing

```bash
python main.py <date> --llm
```

This will:
- Fetch content from the database
- Process with LLM to categorize items
- Update the database with categories and summaries

### Briefing Generation

```bash
python main.py <date> --briefing
```

This will:
- Generate a daily briefing from the categorized content

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
