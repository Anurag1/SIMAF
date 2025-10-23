"""
Test Claude SDK embedding generation - raw output inspection
"""

import asyncio
from claude_agent_sdk import query
from claude_agent_sdk.types import ClaudeAgentOptions

async def test_claude_embedding():
    options = ClaudeAgentOptions(
        system_prompt=(
            "You are an embedding generation system. "
            "Convert text to numerical vectors. "
            "Output ONLY valid JSON with this structure:\n"
            '{"embedding": [float, float, ...]}}\n'
            "The embedding array should contain 1536 float values."
        )
    )

    text = "Hello world"
    prompt = f"Generate embedding for: {text}"

    print("=" * 80)
    print("Testing Claude SDK Embedding Generation")
    print("=" * 80)
    print(f"\nPrompt: {prompt}\n")

    messages = []
    async for message in query(prompt=prompt, options=options):
        messages.append(message)
        print(f"Message type: {type(message).__name__}")
        if hasattr(message, 'content'):
            for block in message.content:
                print(f"  Block type: {type(block).__name__}")
                if hasattr(block, 'text'):
                    print(f"  Text length: {len(block.text)}")
                    print(f"  Text preview: {block.text[:500]}")

    print(f"\n Total messages: {len(messages)}")

if __name__ == "__main__":
    asyncio.run(test_claude_embedding())
