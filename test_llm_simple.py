"""Simple LLM test without retry to see actual errors"""

import os
from dotenv import load_dotenv

load_dotenv()

print("Testing Anthropic...")
print(f"API Key present: {bool(os.getenv('ANTHROPIC_API_KEY'))}")

try:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "What is the Viable System Model in one sentence?"}
        ]
    )

    print(f"✓ Anthropic works: {response.content[0].text}")
except Exception as e:
    print(f"✗ Anthropic failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting Gemini...")
print(f"API Key present: {bool(os.getenv('GOOGLE_API_KEY'))}")

try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    response = model.generate_content("What is the Viable System Model in one sentence?")

    print(f"✓ Gemini works: {response.text}")
except Exception as e:
    print(f"✗ Gemini failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
