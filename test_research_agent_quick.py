"""Quick test of ResearchAgent structure"""

from fractal_agent.agents.research_agent import ResearchAgent
import logging

logging.basicConfig(level=logging.INFO)

print("=" * 80)
print("ResearchAgent Quick Structure Test")
print("=" * 80)
print()

# Test 1: Agent initialization
print("Test 1: Agent Initialization")
print("-" * 80)

agent = ResearchAgent(
    planning_tier="cheap",
    research_tier="balanced",
    synthesis_tier="balanced",
    validation_tier="cheap",
    max_research_questions=1  # Limit to 1 question for speed
)

print("✓ Agent initialized successfully")
print(f"  Planning LM: {agent.planning_lm.tier}")
print(f"  Research LM: {agent.research_lm.tier}")
print(f"  Synthesis LM: {agent.synthesis_lm.tier}")
print(f"  Validation LM: {agent.validation_lm.tier}")
print()

# Test 2: Extract questions
print("Test 2: Question Extraction")
print("-" * 80)

sample_plan = """
1. What is the Viable System Model?
2. How does it apply to organizations?
3. What are the 5 systems?
"""

questions = agent._extract_questions(sample_plan)
print(f"✓ Extracted {len(questions)} questions:")
for i, q in enumerate(questions, 1):
    print(f"  {i}. {q}")
print()

# Test 3: Simple research (very limited)
print("Test 3: Simple Research (1 question only)")
print("-" * 80)

try:
    result = agent.research(
        topic="What is 2+2?",
        verbose=False
    )
    
    print("✓ Research completed successfully")
    print(f"  Topic: {result.topic}")
    print(f"  Questions researched: {result.metadata['num_questions']}")
    print(f"  Total tokens: {result.metadata['total_tokens']}")
    print(f"  Synthesis: {result.synthesis[:100]}...")
    print(f"  Validation: {result.validation[:100]}...")
    
except Exception as e:
    print(f"✗ Research failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("✓ Quick structure test complete!")
print("=" * 80)
