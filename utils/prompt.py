"""
Prompt generation utilities for feeding with the LLM.
"""
import os

ZERO_SHOT_CLSSION_TEMPLATE = None
ZERO_SHOT_EVENT_TEMPLATE = None
FEW_SHOT_EVENT_TEMPLATE = None

prompt_path = os.path.join(os.path.dirname(__file__), 'prompt_templates')

with open(f'{prompt_path}/zero_shot_bin_cls_prompt.txt', 'r', encoding='utf-8') as file:
    ZERO_SHOT_CLSSION_TEMPLATE = file.read()

with open(f'{prompt_path}/zero_shot_event_prompt.txt', 'r', encoding='utf-8') as file:
    ZERO_SHOT_EVENT_TEMPLATE = file.read()

with open(f'{prompt_path}/few_shot_event_prompt.txt', 'r', encoding='utf-8') as file:
    FEW_SHOT_EVENT_TEMPLATE = file.read()

def binary_classificaition_prompt_zero_shot(
        article: str,
        prompt_template: str = ZERO_SHOT_CLSSION_TEMPLATE
    ) -> str:
    """
    Generates a prompt for binary classification of the given article.

    Args:
        article (str): The article to be classified.
        prompt_template (str): The template for the classification prompt.

    Returns:
        str: The formatted prompt for binary classification.
    """
    
    prompt = prompt_template.replace("<article>", article).strip()
    return prompt


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

def event_extraction_prompt_few_shot(
        article: str,
        samples: list,
        prompt_template: str = FEW_SHOT_EVENT_TEMPLATE
    ) -> str:
    """
    Generates a prompt for extracting events from the given text using a few-shot approach.

    Args:
        text (str): The text from which to extract events.

    Returns:
        str: The formatted prompt for event extraction.
    """
    samples_str = ""
    for i, sample in enumerate(samples):
        samples_str += f"Article {i+1}: {sample['text']}\n"
        samples_str += f"Events: {sample['events']}\n\n"
        
    prompt = prompt_template.replace("<samples>", samples_str)
    prompt = prompt.replace("<article>", article).strip()
    return prompt

if __name__ == '__main__':
    import json
    with open('structured_events.json', 'r', encoding='utf-8') as file:
        samples = json.load(file)
    
    print(event_extraction_prompt_few_shot(
        "This is testing article",
        samples=samples
    ))
