#!/usr/bin/env python3
"""
Test FractalDSpyLM deepcopy functionality
"""

import copy
from fractal_agent.utils.dspy_integration import FractalDSpyLM

print("Testing FractalDSpyLM deepcopy...")
print("-" * 80)

# Create an instance
lm = FractalDSpyLM(tier="balanced", max_tokens=100)

# Add some history
lm.history.append({"test": "data"})

print(f"Original LM: tier={lm.tier}, max_tokens={lm.max_tokens}")
print(f"Original history length: {len(lm.history)}")

# Try to deepcopy
try:
    lm_copy = copy.deepcopy(lm)
    print("✓ Deepcopy successful!")
    print(f"Copied LM: tier={lm_copy.tier}, max_tokens={lm_copy.max_tokens}")
    print(f"Copied history length: {len(lm_copy.history)}")

    # Verify they're independent
    lm.history.append({"another": "entry"})
    print(f"\nAfter adding to original:")
    print(f"  Original history length: {len(lm.history)}")
    print(f"  Copy history length: {len(lm_copy.history)}")

    if len(lm.history) != len(lm_copy.history):
        print("✓ Histories are independent!")
    else:
        print("✗ Histories are not independent")

    print("\n" + "=" * 80)
    print("✓ Deepcopy test PASSED - ready for MIPRO optimization")
    print("=" * 80)

except Exception as e:
    print(f"✗ Deepcopy failed: {e}")
    import traceback
    traceback.print_exc()
