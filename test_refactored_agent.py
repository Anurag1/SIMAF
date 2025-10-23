"""Quick test of refactored ResearchAgent"""

from fractal_agent.agents.research_agent import ResearchAgent
from fractal_agent.agents.research_config import ResearchConfig
import logging

logging.basicConfig(level=logging.INFO)

print("=" * 80)
print("Refactored ResearchAgent Test")
print("=" * 80)
print()

# Test 1: Default config (unlimited tokens)
print("Test 1: Default Configuration (unlimited tokens)")
print("-" * 80)

agent = ResearchAgent(max_research_questions=1)

# Call agent instance directly (DSPy Module pattern)
result = agent(topic="What is 2+2?", verbose=False)

print(f"✓ Agent called successfully")
print(f"  Topic: {result.topic}")
print(f"  Synthesis length: {len(result.synthesis)} chars")
print(f"  Total tokens: {result.metadata['total_tokens']}")
print()

# Test 2: Custom config with token limit
print("Test 2: Custom Config with Token Limit")
print("-" * 80)

config = ResearchConfig(
    planning_tier="balanced",
    research_tier="cheap",
    max_tokens=2000  # Optional limit
)

agent2 = ResearchAgent(config=config, max_research_questions=1)
result2 = agent2(topic="What is quantum computing?", verbose=False)

print(f"✓ Agent with custom config called successfully")
print(f"  Config: {config}")
print(f"  Topic: {result2.topic}")
print(f"  Total tokens: {result2.metadata['total_tokens']}")
print()

print("=" * 80)
print("✓ Refactored ResearchAgent working correctly!")
print("=" * 80)
