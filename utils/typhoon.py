import os
from openai import OpenAI
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '.env')
# load the env file
load_dotenv(dotenv_path=env_path)

client = OpenAI(
  base_url="https://api.opentyphoon.ai/v1",
  api_key=os.getenv("TYPHOON_KEY"),
)