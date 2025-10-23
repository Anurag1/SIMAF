"""
Test embedding dimension fix (Gap #6)

Verifies that all embeddings are exactly 1536 dimensions regardless of provider behavior.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from fractal_agent.memory.embeddings import EmbeddingProvider, generate_embedding
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_embedding_dimension_enforcement():
    """Test that embeddings are always 1536 dimensions"""
    print("=" * 80)
    print("Testing Embedding Dimension Fix (Gap #6)")
    print("=" * 80)
    print()

    try:
        # Test 1: Initialize provider
        print("[1/5] Initializing embedding provider...")
        provider = EmbeddingProvider(dimension=1536)
        print(f"✅ Provider: {provider.provider}")
        print(f"   Expected dimension: {provider.dimension}")
        print()

        # Test 2: Single embedding
        print("[2/5] Generating single embedding...")
        text = "Test embedding for dimension validation"
        embedding = provider.generate(text)

        assert len(embedding) == 1536, f"Expected 1536 dims, got {len(embedding)}"
        print(f"✅ Generated embedding: {len(embedding)} dimensions")
        print(f"   First 5 values: {embedding[:5]}")
        print()

        # Test 3: Batch embeddings
        print("[3/5] Generating batch embeddings...")
        texts = [
            "First test text",
            "Second test text",
            "Third test text"
        ]
        embeddings = provider.generate_batch(texts)

        for i, emb in enumerate(embeddings, 1):
            assert len(emb) == 1536, f"Embedding {i}: expected 1536 dims, got {len(emb)}"

        print(f"✅ Generated {len(embeddings)} embeddings, all {embeddings[0].__len__()} dimensions")
        print()

        # Test 4: Convenience function
        print("[4/5] Testing convenience function...")
        embedding = generate_embedding("Convenience function test")

        assert len(embedding) == 1536, f"Expected 1536 dims, got {len(embedding)}"
        print(f"✅ Convenience function: {len(embedding)} dimensions")
        print()

        # Test 5: Multiple calls (consistency check)
        print("[5/5] Testing consistency across multiple calls...")
        embeddings = [generate_embedding("Consistency test") for _ in range(3)]

        dimensions = [len(emb) for emb in embeddings]
        assert all(d == 1536 for d in dimensions), f"Inconsistent dimensions: {dimensions}"
        print(f"✅ All embeddings consistent: {dimensions}")
        print()

        print("=" * 80)
        print("✅ ALL TESTS PASSED - Embedding dimension fix validated!")
        print("=" * 80)
        print()
        print("Summary:")
        print("- All embeddings are exactly 1536 dimensions")
        print("- Dimension enforcement working correctly")
        print("- No dimension mismatch errors")
        print("- Ready for Qdrant insertion")
        print()

        return True

    except AssertionError as e:
        print(f"\n❌ ASSERTION FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_embedding_dimension_enforcement()
    sys.exit(0 if success else 1)
