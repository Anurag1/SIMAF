"""
Test UnifiedLM failover from Anthropic to Gemini

This test verifies that when Anthropic provider fails,
the system correctly falls back to Gemini.
"""

from fractal_agent.utils.llm_provider import UnifiedLM
import logging

# Enable logging to see failover in action
logging.basicConfig(level=logging.INFO)

print("=" * 80)
print("Testing UnifiedLM Failover: Anthropic → Gemini")
print("=" * 80)
print()

# Test 1: Normal operation (Anthropic should succeed)
print("Test 1: Normal operation (Anthropic primary)")
print("-" * 80)
lm = UnifiedLM()

try:
    response = lm(
        prompt="What is 2+2? Answer in one short sentence.",
        max_tokens=50
    )
    print(f"✓ Response: {response['text']}")
    print(f"✓ Provider used: {response['provider']}")
    print(f"✓ Model: {response['model']}")
except Exception as e:
    print(f"✗ Test 1 failed: {e}")

print()
print()

# Test 2: Force Gemini only (to verify Gemini works)
print("Test 2: Gemini-only provider (verify Gemini works)")
print("-" * 80)
lm_gemini = UnifiedLM(
    providers=[
        ("gemini", "gemini-2.0-flash-exp")
    ]
)

try:
    response = lm_gemini(
        prompt="What is 3+3? Answer in one short sentence.",
        max_tokens=50
    )
    print(f"✓ Response: {response['text']}")
    print(f"✓ Provider used: {response['provider']}")
    print(f"✓ Model: {response['model']}")
except Exception as e:
    print(f"✗ Test 2 failed: {e}")

print()
print()

# Test 3: Invalid Anthropic model → should failover to Gemini
print("Test 3: Invalid Anthropic model (should failover to Gemini)")
print("-" * 80)
lm_failover = UnifiedLM(
    providers=[
        ("anthropic", "invalid-model-name"),  # This will fail
        ("gemini", "gemini-2.0-flash-exp")    # Should fallback to this
    ]
)

try:
    response = lm_failover(
        prompt="What is 5+5? Answer in one short sentence.",
        max_tokens=50
    )
    print(f"✓ Response: {response['text']}")
    print(f"✓ Provider used: {response['provider']}")
    print(f"✓ Model: {response['model']}")

    if response['provider'] == 'gemini':
        print("✓ FAILOVER SUCCESSFUL: Correctly fell back to Gemini!")
    else:
        print("✗ FAILOVER FAILED: Should have used Gemini, but used:", response['provider'])

except Exception as e:
    print(f"✗ Test 3 failed: {e}")

print()
print("=" * 80)
print("Failover testing complete!")
print("=" * 80)
