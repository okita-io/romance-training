import os
import sys
from openai import OpenAI

LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LMSTUDIO_API_KEY = "sk-lm-N3auileW:s9AHi8r85ABFOv6Sc4XR"
MODEL_NAME = "openai-gpt-oss-20b-heretic-uncensored-neo-imatrix"

client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)

print("Testing LMStudio connection...")
try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        max_tokens=20,
        temperature=0.0,
    )
    print("Success!")
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print("Error:", e)
