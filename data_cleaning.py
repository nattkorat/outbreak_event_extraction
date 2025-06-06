import os
import json
import argparse
from collections import defaultdict
from utils.cleaning import *


def clean_data(scraped_data: dict, source_type: str='html') -> dict:
    """
    Cleans the scraped data by removing HTML/Markdown tags
    
    Args:
    scraped_data (dict): Dictionary containing scraped data with keys 'html', 'markdown',
    source (str): Source type, either 'html' or 'markdown'.
    
    Returns:
        dict: Dictionary with cleaned text content plus its metadata.
    """
    date = scraped_data['metadata'].get('article:published_time', 'Unknown Date')
    title = scraped_data.get('title', '')
    site_name = scraped_data['metadata'].get('og:site_name', 'Unknown Site')
    source = scraped_data.get('url', '')
    
    if source_type == 'html':
        text = HtmlCleaner.clean_html(scraped_data.get('html', ''))
    else:
        text = MarkdownCleaner.clean_markdown(scraped_data.get('markdown', ''))
    
    cleaned_text = TextCleaner.clean_text(text)
    
    return {
        'content': cleaned_text,
        'date': date,
        'title': title,
        'source': source,
        'site_name': site_name
    }


def collect_and_clean_files(input_dir: str, output_dir: str = 'cleaned_data'):
    """
    Collects JSON files by first and second-level directory, merges them into one JSON list,
    and writes to a new file in the output directory.

    Args:
        input_dir (str): The root directory of the raw JSON files.
        output_dir (str): The directory to store merged output files.
    """

    grouped_files = defaultdict(list)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, input_dir)
            parts = relative_path.split(os.sep)

            if len(parts) < 3:
                continue  # skip if structure is not at least dir1/dir2/file.json

            group_key = (parts[0], parts[1])  # (first_dir, sub_dir)
            grouped_files[group_key].append(full_path)

    for (first_dir, sub_dir), file_list in grouped_files.items():
        combined_data = []

        for file_path in file_list:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    combined_data.append(
                        clean_data(data, source_type='html' if 'html' in data else 'markdown') # clean data based on source type
                    )
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        output_folder = os.path.join(output_dir, first_dir)
        os.makedirs(output_folder, exist_ok=True)

        output_file = os.path.join(output_folder, f"{sub_dir}.json")
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(combined_data, out_f, ensure_ascii=False, indent=3)

        print(f"Cleaned {len(file_list)} files into {output_file}")

def main():
    """
    Runs the data cleaning process by collecting and merging files from the input directory.
    
    """
    argparser = argparse.ArgumentParser(description="Collect and clean JSON files from a directory.")
    argparser.add_argument('--input_dir', type=str, required=True, help="Root directory of the raw JSON files.")
    argparser.add_argument('--output_dir', type=str, default='cleaned_data', help="Directory to store cleaned output files.")
    args = argparser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    print(f"Collecting and cleaning files from {input_dir} into {output_dir}")
    
    # Collect and clean files
    collect_and_clean_files(input_dir, output_dir)



if __name__ == "__main__":
    main()

    
    

    
