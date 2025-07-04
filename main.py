import os
from google import genai
from dotenv import load_dotenv

from utils import prompt
# load the env file
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

example_text = "នៅពេល​កុមារ​មាន​ជំងឺ​រលាក​សួត​រោគសញ្ញា​របស់​ពួកគេ​អាច​ពិបាក​សម្គាល់​។ រោគសញ្ញា​ទាំងនោះ​មានដូចជា​៖ -​ដកដង្ហើម​ញាប់​ -​ពិបាក​ដកដង្ហើម​ -​គ្រុន​ -​ក្អក​ -​ដកដង្ហើម​តឹង លឺសូរ​ក្រេតក្រតៗ​ -​ស្បែក បបូរមាត់ ឬ​ចុង​ម្រាមដៃ មើលទៅ​ពណ៌​ខៀវ​ ​រោគសញ្ញា​នៅក្នុង​ទារក​អាច​មិន​ច្បាស់​ដូចជា​ការ រអាក់រអួល ឬ​ពិបាក​ក្នុងការ​បំបៅ​។ ​ពេលណា​ត្រូវ​ទៅ​ជួប​គ្រូពេទ្យ​? ​សូម​ទៅ​ជួប​គ្រូពេទ្យ​ភ្លាមៗ​ប្រសិនបើ​អ្នក ឬ​កូន​របស់​អ្នកមាន​ជំងឺ​ផ្តាសាយ ឬ​គ្រុនផ្តាសាយ​ដែល​មិន​ធូរស្រាល​ដោយ​ការ​សម្រាក និង​ការព្យាបាល ប្រសិនបើ​រោគសញ្ញា​ចាប់ផ្តើម​កាន់តែ​អាក្រក់​, ប្រសិនបើ​អ្នកមាន​បញ្ហា​សុខភាព​ផ្សេងទៀត ឬ​ប្រព័ន្ធ​ភាពស៊ាំ​ចុះខ្សោយ ឬ​ប្រសិនបើ​អ្នក កត់សម្គាល់​រោគសញ្ញា ដែល​អាច​កើត​មាននៃ​ជំងឺ​រលាក​សួត​។ ​នរណាម្នាក់ ដែលមាន​ការឆ្លង​មេរោគ​សួត​នេះ​ត្រូវការ​ការយកចិត្តទុកដាក់​ខាង​វេជ្ជសាស្ត្រ​៕"
prompt = prompt.event_extraction_prompt_zero_shot(article=example_text)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
)

print(response.text)
