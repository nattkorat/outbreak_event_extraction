from transformers import AutoTokenizer

def get_tokenizer(model_name: str):
    """
    Load a tokenizer for the given model name.

    Args:
        model_name (str): The name of the model for which to load the tokenizer.

    Returns:
        AutoTokenizer: An instance of the tokenizer for the specified model.
    """
    return AutoTokenizer.from_pretrained(model_name)

def count_length(tokenizer, text: str) -> int:
    """
    Count the number of tokens in the given text using the specified tokenizer.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to use for counting tokens.
        text (str): The text to be tokenized.

    Returns:
        int: The number of tokens in the text.
    """
    tokens = tokenizer.encode_plus(text, 
                                   add_special_tokens=False, 
                                   return_tensors='pt')
    return tokens['input_ids'].shape[1]  # Return the number of tokens

if __name__ == "__main__":
    # Example usage
    model_name = "google/gemma-3-1b-it"
    tokenizer = get_tokenizer(model_name)
    
    example_text = "สวัสดีครับ นี่คือข้อความภาษาไทยสำหรับทดสอบการนับโทเค็นโดยใช้โมเดล Gemma 2.0 ของ Google"
    token_count = count_length(tokenizer, example_text)
    
    print(f"Number of tokens in the text: {token_count}")