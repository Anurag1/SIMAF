"""Debug Gemini provider TypeError"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Test 1: No kwargs
print("Test 1: No kwargs")
try:
    response = model.generate_content("What is 2+2?")
    print(f"✓ Success: {response.text}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: With max_tokens (Anthropic parameter)
print("\nTest 2: With max_tokens parameter")
try:
    response = model.generate_content("What is 2+2?", max_tokens=50)
    print(f"✓ Success: {response.text}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")

# Test 3: Correct Gemini parameter
print("\nTest 3: With generation_config")
try:
    response = model.generate_content(
        "What is 2+2?",
        generation_config={"max_output_tokens": 50}
    )
    print(f"✓ Success: {response.text}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")
