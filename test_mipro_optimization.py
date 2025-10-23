#!/usr/bin/env python3
"""
Test MIPRO Optimization Pipeline

This script runs a minimal MIPRO optimization to verify the pipeline works
before running a full optimization.
"""

import logging
logging.basicConfig(level=logging.INFO)

from fractal_agent.agents.optimize_research import optimize_research_agent
from fractal_agent.agents.research_config import ResearchConfig, PresetConfigs

print("=" * 80)
print("MIPRO Optimization Test")
print("=" * 80)
print()
print("This will run a lightweight MIPRO optimization to test the pipeline.")
print("Using minimal settings:")
print("  - auto='light' (MIPRO will determine num_candidates)")
print("  - 1 bootstrapped demo (instead of 3)")
print("  - 2 labeled demos (instead of 4)")
print("  - Default config (NO TOKEN LIMITS)")
print()

# Use default config - NO TOKEN LIMITS
config = ResearchConfig()  # Default: expensive/cheap/balanced/balanced, unlimited tokens
print("Configuration:")
print(config)
print()

print("Starting optimization...")
print("-" * 80)

try:
    optimized_agent = optimize_research_agent(
        config=config,
        use_llm_judge=True,  # Use LLM-as-judge metric
        auto="light",        # Use auto mode (determines candidates automatically)
        max_bootstrapped_demos=1,
        max_labeled_demos=2,
        save_path="test_optimized_agent.json"
    )

    if optimized_agent is None:
        print()
        print("=" * 80)
        print("✗ Optimization returned None")
        print("=" * 80)
    else:
        print()
        print("=" * 80)
        print("✓ Optimization test completed successfully!")
        print("=" * 80)
        print()
        print("Optimized agent saved to: test_optimized_agent.json")

        # Test the optimized agent
        print()
        print("Testing optimized agent...")
        print("-" * 80)

        result = optimized_agent(
            topic="What is prompt caching?",
            verbose=True
        )

        print()
        print("Test result received:")
        print(f"  Synthesis length: {len(result.synthesis)} chars")
        print(f"  Total tokens: {result.metadata['total_tokens']}")

except Exception as e:
    print()
    print("=" * 80)
    print(f"✗ Error during optimization: {e}")
    print("=" * 80)
    import traceback
    traceback.print_exc()
