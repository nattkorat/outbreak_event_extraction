"""Parser module for handling API responses.
"""

import re
import json

def json_string_response_parser(response):
    """Parses a JSON string response and returns a dictionary or list.

    Args:
        response (str): The response possibly containing JSON.

    Returns:
        dict or list: Parsed JSON data.
    """
    try:
        # Try to extract JSON from a ```json fenced code block
        match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            # Fallback: Try to extract any {...} or [...] block (the first one only)
            match = re.search(r'({.*?}|\[.*?\])', response, re.DOTALL)
            if not match:
                raise ValueError("No JSON found in the response.")
            json_str = match.group(1).strip()

        # Try to parse as JSON
        return json.loads(json_str)

    except Exception as e:
        raise ValueError(f"Failed to parse JSON string: {e}")


if __name__ == "__main__":
    # Example usage
    response = '''```json
{
  "Infect": [],
  "Spread": [],
  "Symptom": [],
  "Prevent": [
    {
      "agent": "โรงงานผลิตอาหารสัตว์",
      "disease": "อันตรายที่อาจก่อให้เกิดการเจ็บป่วยหรือบาดเจ็บ",
      "means": "พัฒนาแผนความปลอดภัยด้านอาหารเพื่อป้องกันและลดอันตรายที่อาจก่อให้เกิดการเจ็บป่วยหรือบาดเจ็บให้เหลือน้อยที่สุดอย่างมีนัยสำคัญสำหรับคนและสัตว์",
      "information-source": "FDA",
      "target": "คนและสัตว์",
      "effectiveness": "Not Mentioned"
    },
    {
      "agent": "โรงงานอาหารสัตว์",
      "disease": "อันตรายด้านความปลอดภัยของอาหารที่อาจเกิดขึ้น",
      "means": "ระบุการควบคุมเชิงป้องกันตามความเสี่ยงเพื่อป้องกันหรือลดอันตรายเหล่านั้น และสร้างและดำเนินการตามแผนเพื่อป้องกันไม่ให้อาหารสัตว์ที่ไม่ปลอดภัยเข้าสู่ตลาด",
      "information-source": "FDA",
      "target": "อาหารสัตว์",
      "effectiveness": "Not Mentioned"
    }
  ],
  "Control": [],
  "Cure": [],
  "Death": []
}
```'''
    parsed_response = json_string_response_parser(response)
    print(parsed_response)  # Output: {'key': 'value', 'number': 42}