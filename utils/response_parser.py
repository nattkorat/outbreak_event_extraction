"""Parser module for handling API responses.
"""

import re
import json
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

def json_string_response_parser(text: str) -> dict:
    """
    Extract the first valid JSON object or array from a block of text.

    Args:
        text (str): Input text containing a JSON structure.

    Returns:
        dict or list: Parsed JSON data.
    """
    def find_json_bounds(s: str):
        stack = []
        start = None
        for i, c in enumerate(s):
            if c == '{':
                if start is None:
                    start = i
                stack.append(c)
            elif c == '}':
                if stack:
                    stack.pop()
                    if not stack:
                        return start, i + 1
        return None, None

    # Remove newlines and redundant spaces
    try:
      compact_text = re.sub(r'\s+', ' ', text)
    except Exception as e:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to compact text: {e}")
        compact_text = None, None

    start, end = find_json_bounds(compact_text)
    if start is None or end is None:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} No complete JSON object found.")
        print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} Text was: {text}")
        json_str = "{}" # Fallback to empty JSON if parsing fails
    else:
      json_str = compact_text[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Found a possible JSON block but failed to parse: {e}")

def data_transform(data: dict):
    """
    Transform the parsed JSON data into a structured format.
    Args:
        data (dict): Parsed JSON data.
    Returns:
        list: Transformed data in a structured format.
    """
    transformed_data = []
    for key, value in data.items():
      for e in value:
        temp = {
            "event_type": key,
            "trigger": e['trigger'],
            "arguments": []
        }
        for k, val in e['arguments'].items():
            temp["arguments"].append({
                "role": k,
                "text": val
            })
        transformed_data.append(temp)
    return transformed_data


if __name__ == "__main__":
    # Example usage
    response = '''After carefully reading the news article, I have extracted the relevant information for each event type. Here is the output in JSON format:

{
    "Cure": [
      {
        "trigger": "discharged",
        "arguments": {
          "cured": "Eleven COVID-19 patients",
          "disease": "COVID-19",
          "place": "Oyo",
          "value": "11",
          "information-source": "(url)"
        }
      }
    ]
}

```'''
    parsed_response = json_string_response_parser(response)
    transforming = data_transform(parsed_response)
    print(transforming)  # Output: {'key': 'value', 'number': 42}