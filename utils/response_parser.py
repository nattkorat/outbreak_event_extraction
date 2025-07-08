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


if __name__ == "__main__":
    # Example usage
    response = '''After carefully reading the news article, I have extracted the relevant information for each event type. Here is the output in JSON format:

{
  "Infect": [
    {
      "infected": "កូនជាតិភាគតិច",
      "disease": "ជំងឺរលាក​សួត",
      "place": "ផ្សេងទៀតនៅក្នុងប្រទេសដាំងីឡែន",
      "time": "Not Mentioned",
      "value": "1,000 នាក់",
      "information-source": "មណ្ឌលភាពយន្ត"
    },
    {
      "infected": "គ្រូពេទ្យសាកលវិទ្យាល័យ",
      "disease": "ជំងឺរលាក​សួត",
      "place": "សាកលវិទ្យាល័យភ្នំពេញ",
      "time": "Not Mentioned",
      "value": "500 នាក់",
      "information-source": "មណ្ឌលភាពយន្ត"
    }
  ],
  "Spread": [
    {
      "population": "ប្រជាជន​ផ្សេងទៀតនៅក្នុងប្រទេសដាំងីឡែន",
      "disease": "ជំងឺរលាក​សួត",
      "place": "ផ្សេងទៀតនៅក្នុងប្រទេសដាំងីឡែន",
      "time": "ថ្ងៃ​ទី 15 ខែ ​មករា​ឆ្នាំ 2023",
      "value": "10,000 នាក់",
      "information-source": "មណ្ឌលភាពយន្ត"
    }
  ],
  "Symptom": [
    {
      "person": "គ្រូពេទ្យសាកលវិទ្យាល័យ",
      "symptom": "បន្ថយចំណាត់ហោយទាមរក្សា",
      "disease": "ជំងឺរលាក​សួត",
      "place": "សាកលវិទ្យាល័យភ្នំពេញ",
      "time": "ថ្ងៃ​ទី 10 ខែ ​មករា​ឆ្នាំ 2023",
      "duration": "5 សប្តាហ៍"
    }
  ],
  "Prevent": [
    {
      "agent": "ព្រះរាជច្បាប់ក្លារី",
      "disease": "ជំងឺរលាក​សួត",
      "means": "បញ្ហា វិធីយុទ្ធនាំ",
      "information-source": "ព្រះរាជច្បាប់ក្លារី"
    }
  ],
  "Control": [
    {
      "authority": "សមាគមប្រឹក្សា​អន្តរជាតិ​រួមទាំង ​ព្រះរាជច្បាប់​ក្លារី",
      "disease": "ជំងឺរលាក​សួត",
      "means": "អារម្មណ៍ ហេដ្ឋា",
      "place": "ភ្នំពេញ",
      "time": "ថ្ងៃ​ទី 20 ខែ ​មករា​ឆ្នាំ 2023",
      "information-source": "សារធាតុអន្តរជាតិ"
    }
  ],
  "Cure": [
    {
      "cured": "គ្រូពេទ្យសាកលវិទ្យាល័យ",
      "disease": "ជំងឺរលាក​សួត",
      "means": "គ្រុន",
      "place": "សាកលវិទ្យាល័យភ្នំពេញ",
      "time": "ថ្ងៃ​ទី 25 ខែ ​មករា​ឆ្នាំ 2023",
      "value": "500 នាក់",
      "information-source": "សារធាតុអន្តរជាតិ"
    }
  ],
  "Death": [
    {
      "dead": "មនុស្សដូចជា​កូន​ជាតិភាគតិច",
      "disease": "ជំងឺរលាក​សួត",
      "place": "ផ្សេងទៀតនៅក្នុងប្រទេសដាំងីឡែន",
      "time": "ថ្ងៃ​ទី 20 ខែ ​មករា​ឆ្នាំ 2023",
      "value": "100 នាក់",
      "information-source": "មណ្ឌលភាពយន្ត"
    }
  ]
}

```'''
    parsed_response = json_string_response_parser(response)
    print(parsed_response)  # Output: {'key': 'value', 'number': 42}