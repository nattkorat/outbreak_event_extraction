"""
Prompt generation utilities for feeding with the LLM.
"""
import os

ZERO_SHOT_EVENT_TEMPLATE = None

prompt_path = os.path.join(os.path.dirname(__file__), 'prompt_templates')

with open(f'{prompt_path}/zero_shot_event_prompt.txt', 'r', encoding='utf-8') as file:
    ZERO_SHOT_EVENT_TEMPLATE = file.read()

def classification_prompt_zero_shot(text: str, target_category: list):
    """
    Generates a prompt for classifying text into one of the target categories.

    Args:
        text (str): The text to be classified.
        target_category (list): List of target categories for classification.

    Returns:
        str: The formatted prompt for classification.
    """
    
    categories = ', '.join(target_category)
    
    prompt = f"""Classify the following article into one of these categories: [{categories}], and provide reasoning for your classification.

        Expected output format (json):
        Category: <category_name>
        Confindence: <confidence_score>
        Reasoning: <reasoning for the classification>
        
        Article:\n\n\t {text}
                        
        Category: 
        Confidence:
        Reasoning:"""
    
    return prompt.strip()

def event_extraction_prompt_zero_shot(
        article: str, 
        prompt_template: str = ZERO_SHOT_EVENT_TEMPLATE
    ) -> str:
    """
    Generates a prompt for extracting events from the given text.

    Args:
        text (str): The text from which to extract events.

    Returns:
        str: The formatted prompt for event extraction.
    """
    prompt = prompt_template.replace("<article>", article).strip()
    return prompt


