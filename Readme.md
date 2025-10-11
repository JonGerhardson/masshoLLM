Setup Instructions
1. Prerequisites

    Python 3.8 or newer.

2. Clone the Repository

git clone <your-repository-url>
cd <repository-directory>

3. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

On Linux/macOS:

python3 -m venv venv
source venv/bin/activate

On Windows:

python -m venv venv
.\venv\Scripts\activate

4. Install Dependencies

Install all the required Python libraries from the requirements.txt file.

pip install -r requirements.txt

Configuration

All agent settings are managed in the config.yaml file. Before running the agent, you must configure this file.
LLM Provider Setup

    Choose a Provider: Set the provider field to gemini, openrouter, or lmstudio.

    Add API Keys & Endpoints:

        Gemini: You can either place your key in the config.yaml file OR set it as an environment variable named GEMINI_API_KEY for better security.

        OpenRouter: Add your API key and set the endpoint.

        LMStudio: Set the correct local endpoint for your running LMStudio server.

    Choose a Model: Specify the model you want to use for each provider (e.g., gemini-2.5-flash).

How to Run the Agent
To Process Yesterday's Data

This is the default mode. It will automatically calculate yesterday's date and fetch the corresponding CSV file.

python main.py

To Process a Specific Date

Use the --date command-line argument followed by a date in YYYY-MM-DD format. This is useful for reprocessing old data or catching up on missed days.

python main.py --date 2025-10-09

Project Structure

    main.py: The main entry point and orchestrator of the agent.

    scraper.py: Handles all web scraping, content downloading, and text extraction logic.

    llm_handler.py: Manages all interactions with the configured LLM provider, including single and batch requests.

    database.py: Contains all functions for interacting with the SQLite database.

    config.yaml: The central configuration file for all agent settings.

    requirements.txt: A list of all Python dependencies.

    agent_run.log: A log file that records the agent's operations during a run.

    massgov_updates.db: The SQLite database file where all results are stored.

OutputSetup Instructions
1. Prerequisites

    Python 3.8 or newer.

2. Clone the Repository

git clone <your-repository-url>
cd <repository-directory>

3. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

On Linux/macOS:

python3 -m venv venv
source venv/bin/activate

On Windows:

python -m venv venv
.\venv\Scripts\activate

4. Install Dependencies

Install all the required Python libraries from the requirements.txt file.

pip install -r requirements.txt

Configuration

All agent settings are managed in the config.yaml file. Before running the agent, you must configure this file.
LLM Provider Setup

    Choose a Provider: Set the provider field to gemini, openrouter, or lmstudio.

    Add API Keys & Endpoints:

        Gemini: You can either place your key in the config.yaml file OR set it as an environment variable named GEMINI_API_KEY for better security.

        OpenRouter: Add your API key and set the endpoint.

        LMStudio: Set the correct local endpoint for your running LMStudio server.

    Choose a Model: Specify the model you want to use for each provider (e.g., gemini-2.5-flash).

How to Run the Agent
To Process Yesterday's Data

This is the default mode. It will automatically calculate yesterday's date and fetch the corresponding CSV file.

python main.py

To Process a Specific Date

Use the --date command-line argument followed by a date in YYYY-MM-DD format. This is useful for reprocessing old data or catching up on missed days.

python main.py --date 2025-10-09

Project Structure

    main.py: The main entry point and orchestrator of the agent.

    scraper.py: Handles all web scraping, content downloading, and text extraction logic.

    llm_handler.py: Manages all interactions with the configured LLM provider, including single and batch requests.

    database.py: Contains all functions for interacting with the SQLite database.

    config.yaml: The central configuration file for all agent settings.

    requirements.txt: A list of all Python dependencies.

    agent_run.log: A log file that records the agent's operations during a run.

    massgov_updates.db: The SQLite database file where all results are stored.

OutputSetup Instructions
1. Prerequisites

    Python 3.8 or newer.

2. Clone the Repository

git clone <your-repository-url>
cd <repository-directory>

3. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

On Linux/macOS:

python3 -m venv venv
source venv/bin/activate

On Windows:

python -m venv venv
.\venv\Scripts\activate

4. Install Dependencies

Install all the required Python libraries from the requirements.txt file.

pip install -r requirements.txt

Configuration

All agent settings are managed in the config.yaml file. Before running the agent, you must configure this file.
LLM Provider Setup

    Choose a Provider: Set the provider field to gemini, openrouter, or lmstudio.

    Add API Keys & Endpoints:

        Gemini: You can either place your key in the config.yaml file OR set it as an environment variable named GEMINI_API_KEY for better security.

        OpenRouter: Add your API key and set the endpoint.

        LMStudio: Set the correct local endpoint for your running LMStudio server.

    Choose a Model: Specify the model you want to use for each provider (e.g., gemini-2.5-flash).

How to Run the Agent
To Process Yesterday's Data

This is the default mode. It will automatically calculate yesterday's date and fetch the corresponding CSV file.

python main.py

To Process a Specific Date

Use the --date command-line argument followed by a date in YYYY-MM-DD format. This is useful for reprocessing old data or catching up on missed days.

python main.py --date 2025-10-09

Project Structure

    main.py: The main entry point and orchestrator of the agent.

    scraper.py: Handles all web scraping, content downloading, and text extraction logic.

    llm_handler.py: Manages all interactions with the configured LLM provider, including single and batch requests.

    database.py: Contains all functions for interacting with the SQLite database.

    config.yaml: The central configuration file for all agent settings.

    requirements.txt: A list of all Python dependencies.

    agent_run.log: A log file that records the agent's operations during a run.

    massgov_updates.db: The SQLite database file where all results are stored.

Output

The agent's output is stored in the SQLite database file (massgov_updates.db). Each day's run is saved in a separate table named massgov_YYYY_MM_DD.

The table has the following columns:

    url: The original URL from the sitemap.

    lastmodified: The lastmod value from the sitemap CSV.

    filetype: The detected file type (e.g., HTML, PDF, DOCX).

    page_date: The publication or "last updated" date found on the page, if any.

    is_new: The agent's determination ('yes', 'no', or 'maybe').

    summary: The two-sentence summary

The agent's output is stored in the SQLite database file (massgov_updates.db). Each day's run is saved in a separate table named massgov_YYYY_MM_DD.

The table has the following columns:

    url: The original URL from the sitemap.

    lastmodified: The lastmod value from the sitemap CSV.

    filetype: The detected file type (e.g., HTML, PDF, DOCX).

    page_date: The publication or "last updated" date found on the page, if any.

    is_new: The agent's determination ('yes', 'no', or 'maybe').

    summary: The two-sentence summary

The agent's output is stored in the SQLite database file (massgov_updates.db). Each day's run is saved in a separate table named massgov_YYYY_MM_DD.

The table has the following columns:

    url: The original URL from the sitemap.

    lastmodified: The lastmod value from the sitemap CSV.

    filetype: The detected file type (e.g., HTML, PDF, DOCX).

    page_date: The publication or "last updated" date found on the page, if any.

    is_new: The agent's determination ('yes', 'no', or 'maybe').

    summary: The two-sentence summary
