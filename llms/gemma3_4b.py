import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-3-4b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

def chat_with_gemma3_4b(user_input: str) -> str:
    system_prompt = "You are a helpful assistant."
    prompt = (
        "<start_of_turn>system\n" + system_prompt + "<end_of_turn>\n" +
        "<start_of_turn>user\n" + user_input + "<end_of_turn>\n" +
        "<start_of_turn>model\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.convert_tokens_to_ids("<end_of_turn>")
    )

    # Decode and clean the output
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = decoded.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
    return response



if __name__ == "__main__":
    # Example usage
    from utils.prompt import binary_classificaition_prompt_zero_shot
    
    prompt = binary_classificaition_prompt_zero_shot(
        "នៅពេល​កុមារ​មាន​ជំងឺ​រលាក​សួត​រោគសញ្ញា​របស់​ពួកគេ​អាច​ពិបាក​សម្គាល់​។ រោគសញ្ញា​ទាំងនោះ​មានដូចជា​៖ -​ដកដង្ហើម​ញាប់​ -​ពិបាក​ដកដង្ហើម​ -​គ្រុន​ -​ក្អក​ -​ដកដង្ហើម​តឹង លឺសូរ​ក្រេតក្រតៗ​ -​ស្បែក បបូរមាត់ ឬ​ចុង​ម្រាមដៃ មើលទៅ​ពណ៌​ខៀវ​ ​រោគសញ្ញា​នៅក្នុង​ទារក​អាច​មិន​ច្បាស់​ដូចជា​ការ រអាក់រអួល ឬ​ពិបាក​ក្នុងការ​បំបៅ​។ ​ពេលណា​ត្រូវ​ទៅ​ជួប​គ្រូពេទ្យ​? ​សូម​ទៅ​ជួប​គ្រូពេទ្យ​ភ្លាមៗ​ប្រសិនបើ​អ្នក ឬ​កូន​របស់​អ្នកមាន​ជំងឺ​ផ្តាសាយ ឬ​គ្រុនផ្តាសាយ​ដែល​មិន​ធូរស្រាល​ដោយ​ការ​សម្រាក និង​ការព្យាបាល ប្រសិនបើ​រោគសញ្ញា​ចាប់ផ្តើម​កាន់តែ​អាក្រក់​, ប្រសិនបើ​អ្នកមាន​បញ្ហា​សុខភាព​ផ្សេងទៀត ឬ​ប្រព័ន្ធ​ភាពស៊ាំ​ចុះខ្សោយ ឬ​ប្រសិនបើ​អ្នក កត់សម្គាល់​រោគសញ្ញា ដែល​អាច​កើត​មាននៃ​ជំងឺ​រលាក​សួត​។ ​នរណាម្នាក់ ដែលមាន​ការឆ្លង​មេរោគ​សួត​នេះ​ត្រូវការ​ការយកចិត្តទុកដាក់​ខាង​វេជ្ជសាស្ត្រ​៕"
    )
    response = chat_with_gemma3_4b(prompt)
    
    print(f"Response from Gemma 3.4B: {response}")
