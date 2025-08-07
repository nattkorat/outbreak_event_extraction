import os
from openai import OpenAI
from dotenv import load_dotenv

# load the env file
load_dotenv()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPEN_ROUTER_KEY"),
)