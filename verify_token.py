import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
API_BASE_URL = "https://router.huggingface.co/v1"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

print(f"Testing token: {HF_TOKEN[:8]}...{HF_TOKEN[-4:]}")

try:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Hello, are you working?"}],
        max_tokens=10
    )
    print("SUCCESS: Token is valid and working!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"FAILURE: Token or API error: {e}")
