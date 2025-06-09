"""
Prompt generation utilities for feeding with the LLM.
"""

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
    
    prompt = f"""Classify the following text into one of these categories: [{categories}]
        Text: {text}
                        
        Category: """
    
    return prompt.strip()

if __name__ == "__main__":
    # Example usage
    example_text = "This is a sample text that needs to be classified."
    example_categories = ["outbreak", "health promotion", "case report", "none health topic"]
    
    prompt = classification_prompt(example_text, example_categories)
    print(prompt)
