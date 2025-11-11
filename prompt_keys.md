Prompt Template Variables

This file documents the template variables used in the prompts/ directory.

1_parse_meeting.txt

[CURRENT_DATE]: The current date (YYYY-MM-DD) when the script is run.

[MEETING_URL]: The source URL of the meeting announcement.

[MEETING_TEXT]: The raw, extracted text from the meeting page.

2_summarize_news.txt

[CURRENT_DATE]: The current date (YYYY-MM-DD).

[NEWS_URL]: The source URL of the news item.

[NEWS_TEXT]: The raw, extracted text from the news page.

Note: This text may contain a [NOTE TO EDITOR: ...] tag if it was truncated.

3_select_top_stories.txt

[CURRENT_DATE]: The current date (YYYY-MM-DD).

[NEWS_SUMMARIES_LIST]: A block of text containing all generated news summaries, each prefixed with its [SOURCE: <url>].

4_format_final_briefing.txt

[CURRENT_DATE]: The current date (YYYY-MM-DD).

[TARGET_DATE]: The date the report is for (YYYY-MM-DD), passed via the command line.

[TOP_STORY_URLS_LIST]: A simple list of the URLs identified as top stories.

[ALL_NEWS_SUMMARIES_LIST]: A block of text containing all generated news summaries, each prefixed with its [SOURCE: <url>].

[MEETING_JSON_LIST]: A block of text containing the structured JSON objects for all upcoming meetings.
