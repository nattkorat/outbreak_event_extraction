from ollama import chat, ChatResponse

def chat_with_ollama(prompt: str, model: str = 'gemma3:1b') -> str: # This is the resource-intensive model, cannot run with many requests
    """
    Interacts with the Ollama architecture model to generate a response based on the provided prompt.

    Args:
        prompt (str): The input prompt for the model.
        model (str): The name of the model to use.

    Returns:
        str: The generated response from the model.
    """
    response: ChatResponse = chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=False  
    )
    
    return response.message.content.strip()
