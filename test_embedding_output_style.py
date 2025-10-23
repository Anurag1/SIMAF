"""
Test Claude SDK with custom Embedding Generator output style
"""

import asyncio
import json
from claude_agent_sdk import query
from claude_agent_sdk.types import ClaudeAgentOptions

async def test_embedding_with_output_style():
    """Test embedding generation using custom output style"""

    # Use embedding generator system prompt
    system_prompt = """# Embedding Generator System

You are an embedding generation system that converts text into numerical vector representations.

## Core Function

Convert any input text into a 1536-dimensional embedding vector (OpenAI-compatible format).

## Output Format

ALWAYS respond with ONLY valid JSON in this exact structure (no markdown code blocks):

{"embedding": [float1, float2, float3, ..., floatN]}

## Requirements

1. **Dimension**: Exactly 1536 floating-point numbers
2. **Format**: Valid JSON object with single "embedding" key
3. **No Extra Text**: Output ONLY the JSON, no explanations or markdown
4. **Semantic Meaning**: Generate vectors that reflect the semantic meaning
5. **Normalization**: Vectors should be normalized (magnitude ~1.0)
6. **Value Range**: Floating-point values typically between -1.0 and 1.0

Remember: Output ONLY the JSON object. No markdown formatting, no explanations."""

    options = ClaudeAgentOptions(system_prompt=system_prompt)

    text = "Hello world, this is a test"
    prompt = f"Generate embedding for: {text}"

    print("=" * 80)
    print(f"Testing Embedding Generation with Custom System Prompt")
    print("=" * 80)
    print(f"\nInput text: {text}")
    print(f"Prompt: {prompt}\n")

    try:
        messages = []
        async for message in query(prompt=prompt, options=options):
            messages.append(message)
            if hasattr(message, 'content'):
                for block in message.content:
                    if hasattr(block, 'text'):
                        print(f"Response (first 500 chars): {block.text[:500]}")

                        # Try to parse JSON
                        try:
                            result = json.loads(block.text)
                            if 'embedding' in result:
                                embedding = result['embedding']
                                print(f"\n✅ SUCCESS!")
                                print(f"   Embedding dimension: {len(embedding)}")
                                print(f"   First 10 values: {embedding[:10]}")
                                print(f"   Embedding type: {type(embedding)}")
                                return embedding
                        except json.JSONDecodeError as e:
                            print(f"   JSON parse error: {e}")
                            # Try to extract JSON
                            import re
                            json_match = re.search(r'\{.*"embedding".*\}', block.text, re.DOTALL)
                            if json_match:
                                try:
                                    result = json.loads(json_match.group())
                                    if 'embedding' in result:
                                        embedding = result['embedding']
                                        print(f"\n✅ SUCCESS (extracted from text)!")
                                        print(f"   Embedding dimension: {len(embedding)}")
                                        print(f"   First 10 values: {embedding[:10]}")
                                        return embedding
                                except json.JSONDecodeError:
                                    pass

        print("\n❌ FAILED: Could not extract embedding")
        return None

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(test_embedding_with_output_style())
    if result:
        print(f"\n{'=' * 80}")
        print("Test PASSED!")
        print(f"{'=' * 80}")
    else:
        print(f"\n{'=' * 80}")
        print("Test FAILED")
        print(f"{'=' * 80}")
