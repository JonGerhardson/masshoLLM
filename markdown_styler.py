import markdown
import argparse
import sys

def convert_markdown_to_html(input_file, output_file):
    """
    Reads a Markdown file, converts it to HTML, and wraps it in a
    styled HTML structure that mimics a specific text editor theme.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            md_text = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{input_file}': {e}", file=sys.stderr)
        sys.exit(1)

    # Convert markdown text to HTML body content
    html_content = markdown.markdown(md_text)

    # Define the CSS to mimic the text editor's style from the image
    # This is based on the Solarized Light theme, which appears to match.
    css_style = """
    <style>
        body {
            background-color: #fdf6e3; /* Creamy background */
            font-family: Arial, sans-serif; /* Set font to Arial */
            color: #586e75; /* Darker main text for better readability */
            line-height: 1.6;
            margin: 0;
            padding: 2em;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: #eee8d5; /* Slightly darker container bg */
            padding: 2em;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        h1, h2, h3, h4, h5, h6 {
            color: #d33682; /* Pink/magenta for headers */
            border-bottom: 1px solid #eee8d5;
            padding-bottom: 5px;
        }
        strong, b {
            color: #cb4b16; /* Orange for bold text */
            font-weight: bold;
        }
        a {
            color: #005b99; /* Darker blue for links for better readability */
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        ul, ol {
            padding-left: 20px;
        }
        li {
            margin-bottom: 0.5em;
        }
        code, pre {
            font-family: 'Courier New', Courier, monospace; /* Keep code blocks monospaced */
        }
        code {
            background-color: #eee8d5;
            padding: 2px 5px;
            border-radius: 4px;
            color: #b58900; /* Yellow for code */
        }
        pre {
            background-color: #eee8d5;
            padding: 1em;
            border-radius: 4px;
            overflow-x: auto;
        }
        hr {
            border: 0;
            border-top: 1px solid #93a1a1; /* Line color for horizontal rule */
            margin: 2em 0;
        }
    </style>
    """

    # Full HTML document template
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{output_file}</title>
        {css_style}
    </head>
    <body>
        <div class="container">
            {html_content}
        </div>
    </body>
    </html>
    """

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"Successfully converted '{input_file}' to '{output_file}'")
    except Exception as e:
        print(f"Error writing to file '{output_file}': {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert a Markdown file to a styled HTML file that looks like a text editor theme."
    )
    parser.add_argument("input_file", help="The path to the input Markdown file.")
    parser.add_argument("output_file", help="The path for the output HTML file.")
    
    args = parser.parse_args()
    
    convert_markdown_to_html(args.input_file, args.output_file)

