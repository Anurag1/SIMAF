"""
Simple test to isolate KnowledgeExtractionAgent initialization issue.

Tests just the agent initialization without GraphRAG or full pipeline.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("Simple Knowledge Extraction Agent Test")
print("=" * 80)
print()

try:
    print("[1/3] Importing KnowledgeExtractionAgent...")
    from fractal_agent.agents.knowledge_extraction_agent import KnowledgeExtractionAgent
    print("✅ Import successful")
    print()

    print("[2/3] Initializing agent...")
    agent = KnowledgeExtractionAgent()
    print("✅ Agent initialized")
    print(f"   Agent type: {type(agent)}")
    print(f"   Has extract module: {hasattr(agent, 'extract')}")
    print()

    print("[3/3] Testing basic extraction (simple text)...")
    result = agent(
        task_description="Test task",
        task_output="This is a simple test output with basic content."
    )

    print("✅ Extraction completed")
    print(f"   Entities: {len(result['entities'])}")
    print(f"   Relationships: {len(result['relationships'])}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print()

    print("=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print()
    import traceback
    traceback.print_exc()
    print()
    print("=" * 80)
    print("TESTS FAILED")
    print("=" * 80)
    sys.exit(1)
