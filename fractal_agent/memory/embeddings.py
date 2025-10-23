"""
Embedding Generation - Phase 5 (Fixed)

Unified embedding generation using proper embedding models.

DEFAULT PROVIDER: sentence-transformers (local, no API keys, consistent results)
FALLBACK: Claude SDK (not recommended for embeddings - LLMs generate text, not vectors)

IMPORTANT: LLMs like Claude are NOT designed for numerical embedding generation.
They produce inconsistent dimensions and poor quality embeddings. Always use
proper embedding models like sentence-transformers for production use.

Author: BMad
Date: 2025-10-22
"""

from typing import List, Optional
import os
import logging
import json
import asyncio

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """
    Unified embedding provider using proper embedding models.

    DEFAULT: sentence-transformers (local, no API keys, consistent embeddings)
    FALLBACK: Claude SDK (not recommended - LLMs generate text, not vectors)

    Usage:
        >>> # Default: sentence-transformers (recommended)
        >>> provider = EmbeddingProvider()
        >>> embedding = provider.generate("Hello world")
        >>> len(embedding)
        1536  # OpenAI-compatible format (768-dim model padded to 1536)

        >>> # Batch processing
        >>> embeddings = provider.generate_batch(["Hello", "World"])
        >>> len(embeddings)
        2

    Attributes:
        provider: Name of current provider ("sentence-transformers" or "claude-sdk")
        dimension: Embedding vector dimension (1536 for OpenAI compatibility)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        dimension: int = 1536
    ):
        """
        Initialize embedding provider.

        Args:
            provider: Specific provider to use. Options:
                     - "sentence-transformers" (default, recommended for proper embeddings)
                     - "claude-sdk" (fallback, not recommended - LLMs generate text not vectors)
                     If None, auto-selects best available provider
            dimension: Embedding dimension (default: 1536 for OpenAI compatibility)

        Raises:
            ValueError: If provider initialization fails
        """
        self.dimension = dimension
        self.provider = provider or self._select_provider()

        if self.provider == "claude-sdk":
            self._init_claude_sdk()
        elif self.provider == "sentence-transformers":
            self._init_sentence_transformers()
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

        logger.info(
            f"Initialized EmbeddingProvider: {self.provider} (dim: {self.dimension})"
        )

    def _select_provider(self) -> str:
        """
        Auto-select provider based on available resources.

        Priority:
        1. sentence-transformers (proper embedding model, consistent results)
        2. claude-sdk (fallback, but not recommended for embeddings)

        NOTE: LLMs like Claude are NOT designed for numerical embedding generation.
        They generate text, not consistent numerical vectors. sentence-transformers
        uses actual embedding models trained specifically for this task.
        """
        # Always prefer sentence-transformers (proper embedding model)
        try:
            import sentence_transformers
            logger.info("Using sentence-transformers for embeddings (recommended)")
            return "sentence-transformers"
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to claude-sdk (not recommended for embeddings)")

        # Fallback to Claude SDK (not ideal, but better than nothing)
        try:
            import claude_agent_sdk
            logger.warning("Using Claude SDK for embeddings - this may produce inconsistent dimensions")
            return "claude-sdk"
        except ImportError:
            raise ValueError(
                "No embedding provider available. "
                "Install sentence-transformers (recommended) or claude-agent-sdk (fallback)"
            )

    def _init_claude_sdk(self):
        """Initialize Claude SDK embedding provider."""
        try:
            from claude_agent_sdk import query
            from claude_agent_sdk.types import ClaudeAgentOptions
            self.query = query

            # Custom system prompt for embedding generation in OpenAI-compatible format
            system_prompt = f"""# Embedding Generator System

You are an embedding generation system that converts text into numerical vector representations.

## Core Function

Convert any input text into a {self.dimension}-dimensional embedding vector (OpenAI-compatible format).

## Output Format

ALWAYS respond with ONLY valid JSON in this exact structure (no markdown code blocks):

{{"embedding": [float1, float2, float3, ..., float{self.dimension}]}}

## Critical Requirements

1. **Dimension**: EXACTLY {self.dimension} floating-point numbers (count them!)
2. **Format**: Valid JSON object with single "embedding" key
3. **No Extra Text**: Output ONLY the JSON, no explanations, no markdown blocks
4. **Semantic Meaning**: Generate vectors that reflect the semantic meaning
   - Similar texts should have similar vector representations
   - Different topics should have distinct vector patterns
5. **Normalization**: Vectors should be normalized (magnitude ~1.0)
6. **Consistency**: Same text should generate the same embedding
7. **Value Range**: Floating-point values between -1.0 and 1.0

## Vector Generation Strategy

Generate semantically meaningful embeddings by:
- Analyzing key concepts and entities in the text
- Considering semantic relationships and context
- Distributing information across ALL {self.dimension} dimensions
- Maintaining consistency for similar content
- Creating distinct patterns for different topics

Remember: Output ONLY the JSON object with exactly {self.dimension} values. No markdown, no explanations."""

            self.options = ClaudeAgentOptions(system_prompt=system_prompt)
            logger.debug("Initialized Claude SDK embedding provider with custom system prompt")
        except ImportError:
            raise ImportError(
                "claude-agent-sdk not installed. "
                "This should not happen as it's checked in _select_provider"
            )

    def _init_sentence_transformers(self):
        """Initialize sentence-transformers (local embeddings, no API key needed)."""
        try:
            from sentence_transformers import SentenceTransformer

            # Use model that matches our dimension
            if self.dimension == 384:
                model_name = "all-MiniLM-L6-v2"
            elif self.dimension == 768:
                model_name = "all-mpnet-base-v2"
            elif self.dimension == 1536:
                # For 1536-dim compatibility, we'll pad the 768-dim output
                model_name = "all-mpnet-base-v2"
                logger.info("Using 768-dim model with padding for 1536-dim compatibility")
            else:
                model_name = "all-MiniLM-L6-v2"
                self.dimension = 384
                logger.warning(f"Unsupported dimension, using 384-dim model")

            self.client = SentenceTransformer(model_name)
            self.base_dimension = self.client.get_sentence_embedding_dimension()

            logger.debug(f"Initialized sentence-transformers: {model_name}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

    def _enforce_dimension(self, embedding: List[float]) -> List[float]:
        """
        Enforce embedding dimension by padding or truncating.

        This is CRITICAL for Qdrant compatibility - all embeddings MUST be exactly
        self.dimension length or Qdrant will reject them.

        Args:
            embedding: Raw embedding vector

        Returns:
            Embedding adjusted to exactly self.dimension length

        Raises:
            ValueError: If embedding is empty or dimension is severely wrong
        """
        actual_dim = len(embedding)

        if actual_dim == self.dimension:
            # Perfect - no adjustment needed
            return embedding

        if actual_dim == 0:
            raise ValueError("Cannot enforce dimension on empty embedding")

        # Check if dimension is severely wrong (>50% off)
        dimension_ratio = actual_dim / self.dimension
        if dimension_ratio < 0.5 or dimension_ratio > 1.5:
            logger.error(
                f"SEVERE dimension mismatch: got {actual_dim}, expected {self.dimension} "
                f"(ratio: {dimension_ratio:.2f})"
            )
            raise ValueError(
                f"Embedding dimension severely wrong: expected {self.dimension}, got {actual_dim}. "
                "This suggests a provider configuration error."
            )

        if actual_dim < self.dimension:
            # Pad with zeros
            padding_needed = self.dimension - actual_dim
            logger.warning(
                f"Embedding too short ({actual_dim}), padding with {padding_needed} zeros to reach {self.dimension}"
            )
            embedding = embedding + [0.0] * padding_needed

        elif actual_dim > self.dimension:
            # Truncate
            extra_dims = actual_dim - self.dimension
            logger.warning(
                f"Embedding too long ({actual_dim}), truncating {extra_dims} dimensions to reach {self.dimension}"
            )
            embedding = embedding[:self.dimension]

        # Final validation
        assert len(embedding) == self.dimension, \
            f"Dimension enforcement failed: expected {self.dimension}, got {len(embedding)}"

        return embedding

    def generate(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector (guaranteed to be self.dimension length)

        Raises:
            ValueError: If text is empty
            Exception: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")

        try:
            if self.provider == "claude-sdk":
                embedding = self._generate_claude_sdk(text)
            elif self.provider == "sentence-transformers":
                embedding = self._generate_sentence_transformers(text)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            # CRITICAL: Validate and enforce dimension
            embedding = self._enforce_dimension(embedding)

            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed with {self.provider}: {e}")
            raise

    def _generate_claude_sdk(self, text: str) -> List[float]:
        """Generate embedding using Claude SDK.

        Note: Returns raw embedding - dimension enforcement handled by generate().
        """
        import re

        async def _async_generate():
            prompt = f"Generate embedding for this text:\n\n{text}"

            # Collect all messages from Claude
            messages = []
            async for message in self.query(prompt=prompt, options=self.options):
                messages.append(message)

            # Extract embedding from the last assistant message
            for message in reversed(messages):
                if hasattr(message, 'content'):
                    for block in message.content:
                        if hasattr(block, 'text'):
                            response_text = block.text.strip()

                            # Remove markdown code blocks if present (```json ... ```)
                            if response_text.startswith('```'):
                                code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
                                if code_block_match:
                                    response_text = code_block_match.group(1).strip()

                            try:
                                # Try to parse JSON from Claude's response
                                result = json.loads(response_text)
                                if 'embedding' in result:
                                    embedding = result['embedding']
                                    logger.debug(f"Claude SDK generated {len(embedding)}-dim embedding")
                                    return embedding
                            except json.JSONDecodeError:
                                # Try to extract JSON from text (in case of explanation text)
                                json_match = re.search(r'\{.*"embedding".*\}', response_text, re.DOTALL)
                                if json_match:
                                    try:
                                        result = json.loads(json_match.group())
                                        if 'embedding' in result:
                                            embedding = result['embedding']
                                            logger.debug(f"Extracted embedding from text: {len(embedding)} dimensions")
                                            return embedding
                                    except json.JSONDecodeError:
                                        continue

            raise ValueError("Could not extract embedding from Claude response")

        # Run async function
        return asyncio.run(_async_generate())

    def _generate_sentence_transformers(self, text: str) -> List[float]:
        """Generate embedding using sentence-transformers.

        Note: Returns raw embedding - dimension enforcement handled by generate().
        """
        embedding = self.client.encode(text, convert_to_numpy=True).tolist()
        logger.debug(f"Sentence-transformers generated {len(embedding)}-dim embedding")
        return embedding

    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).

        All returned embeddings are guaranteed to be exactly self.dimension length.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors (each guaranteed to be self.dimension length)

        Raises:
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Cannot generate embeddings for empty list")

        logger.debug(f"Generating batch embeddings for {len(texts)} texts")

        if self.provider == "claude-sdk":
            # For Claude SDK, process sequentially (dimension enforcement automatic via generate())
            return [self.generate(text) for text in texts]
        elif self.provider == "sentence-transformers":
            # sentence-transformers supports efficient batching
            embeddings = self.client.encode(texts, convert_to_numpy=True).tolist()

            # Enforce dimension on all embeddings
            result = []
            for embedding in embeddings:
                enforced = self._enforce_dimension(embedding)
                result.append(enforced)

            return result


# Convenience function for simple use cases
_default_provider: Optional[EmbeddingProvider] = None


def generate_embedding(text: str, provider: Optional[str] = None) -> List[float]:
    """
    Generate embedding vector for text (convenience function).

    Uses a singleton EmbeddingProvider instance for efficiency.

    Args:
        text: Input text to embed
        provider: Optional provider override ("claude-sdk" or "sentence-transformers")

    Returns:
        List of floats representing the embedding vector

    Usage:
        >>> embedding = generate_embedding("Hello world")
        >>> len(embedding)
        1536
    """
    global _default_provider

    # Initialize default provider if needed
    if _default_provider is None or (provider and provider != _default_provider.provider):
        _default_provider = EmbeddingProvider(provider=provider)

    return _default_provider.generate(text)


def generate_embeddings_batch(
    texts: List[str],
    provider: Optional[str] = None
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts (batch processing).

    Args:
        texts: List of input texts
        provider: Optional provider override

    Returns:
        List of embedding vectors

    Usage:
        >>> texts = ["Hello", "World", "AI"]
        >>> embeddings = generate_embeddings_batch(texts)
        >>> len(embeddings)
        3
    """
    global _default_provider

    # Initialize default provider if needed
    if _default_provider is None or (provider and provider != _default_provider.provider):
        _default_provider = EmbeddingProvider(provider=provider)

    return _default_provider.generate_batch(texts)


# Demo
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Embedding Generation Demo - Phase 3 (Claude SDK)")
    print("=" * 80)
    print()

    try:
        # Test 1: Initialize provider
        print("[1/4] Initializing embedding provider...")
        provider = EmbeddingProvider()
        print(f"✅ Using provider: {provider.provider}")
        print(f"   Embedding dimension: {provider.dimension}")
        print()

        # Test 2: Single embedding
        print("[2/4] Generating single embedding...")
        text = "ResearchAgent produces high-quality research reports"
        embedding = provider.generate(text)
        print(f"✅ Generated embedding for: '{text}'")
        print(f"   Dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        print()

        # Test 3: Batch embeddings
        print("[3/4] Generating batch embeddings...")
        texts = [
            "IntelligenceAgent analyzes performance metrics",
            "ControlAgent coordinates multi-agent workflows",
            "GraphRAG stores knowledge with temporal validity"
        ]
        embeddings = provider.generate_batch(texts)
        print(f"✅ Generated {len(embeddings)} embeddings")
        for i, (text, emb) in enumerate(zip(texts, embeddings), 1):
            print(f"   {i}. '{text[:40]}...' -> {len(emb)} dims")
        print()

        # Test 4: Convenience function
        print("[4/4] Testing convenience function...")
        embedding = generate_embedding("Knowledge graph with embeddings")
        print(f"✅ Convenience function works: {len(embedding)} dimensions")
        print()

        print("=" * 80)
        print("Embedding Generation Demo Complete!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: Claude SDK embeddings require claude-agent-sdk")
        print("Fallback to sentence-transformers if Claude SDK unavailable")
