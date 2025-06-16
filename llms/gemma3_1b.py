from ollama import chat, ChatResponse

def chat_with_gemma3_1b(prompt: str, model: str = 'gemma3:1b') -> str: # This is the resource-intensive model, cannot run with many requests
    """
    Interacts with the Gemma 3.1B model to generate a response based on the provided prompt.

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

if __name__ == "__main__":
    from utils.prompt import binary_classificaition_prompt_zero_shot # need to execute by: `python -m llms.gemma3_1b`
    # Example usage
    example_prompt = "នៅពេល​កុមារ​មាន​ជំងឺ​រលាក​សួត​រោគសញ្ញា​របស់​ពួកគេ​អាច​ពិបាក​សម្គាល់​។ រោគសញ្ញា​ទាំងនោះ​មានដូចជា​៖ -​ដកដង្ហើម​ញាប់​ -​ពិបាក​ដកដង្ហើម​ -​គ្រុន​ -​ក្អក​ -​ដកដង្ហើម​តឹង លឺសូរ​ក្រេតក្រតៗ​ -​ស្បែក បបូរមាត់ ឬ​ចុង​ម្រាមដៃ មើលទៅ​ពណ៌​ខៀវ​ ​រោគសញ្ញា​នៅក្នុង​ទារក​អាច​មិន​ច្បាស់​ដូចជា​ការ រអាក់រអួល ឬ​ពិបាក​ក្នុងការ​បំបៅ​។ ​ពេលណា​ត្រូវ​ទៅ​ជួប​គ្រូពេទ្យ​? ​សូម​ទៅ​ជួប​គ្រូពេទ្យ​ភ្លាមៗ​ប្រសិនបើ​អ្នក ឬ​កូន​របស់​អ្នកមាន​ជំងឺ​ផ្តាសាយ ឬ​គ្រុនផ្តាសាយ​ដែល​មិន​ធូរស្រាល​ដោយ​ការ​សម្រាក និង​ការព្យាបាល ប្រសិនបើ​រោគសញ្ញា​ចាប់ផ្តើម​កាន់តែ​អាក្រក់​, ប្រសិនបើ​អ្នកមាន​បញ្ហា​សុខភាព​ផ្សេងទៀត ឬ​ប្រព័ន្ធ​ភាពស៊ាំ​ចុះខ្សោយ ឬ​ប្រសិនបើ​អ្នក កត់សម្គាល់​រោគសញ្ញា ដែល​អាច​កើត​មាននៃ​ជំងឺ​រលាក​សួត​។ ​នរណាម្នាក់ ដែលមាន​ការឆ្លង​មេរោគ​សួត​នេះ​ត្រូវការ​ការយកចិត្តទុកដាក់​ខាង​វេជ្ជសាស្ត្រ​៕"
    example_prompt = binary_classificaition_prompt_zero_shot(example_prompt)
    response = chat_with_gemma3_1b(example_prompt)
    print(response)  # Output the response from the model
