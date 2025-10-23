"""
Training Examples for ResearchAgent DSPy Optimization

Provides training dataset in DSPy format for MIPRO optimization.
Each example specifies a research topic and expected key aspects to cover.

Author: BMad
Date: 2025-10-18
"""

import dspy


# ============================================================================
# Training Examples
# ============================================================================

def get_training_examples():
    """
    Get training examples for ResearchAgent optimization.

    Each example has:
    - topic: The research question
    - key_aspects: List of important aspects that should be covered

    Returns:
        List of dspy.Example objects
    """
    examples = [
        dspy.Example(
            topic="What is the Viable System Model?",
            key_aspects=[
                "Stafford Beer",
                "cybernetics",
                "five systems",
                "recursion",
                "organizational viability"
            ]
        ).with_inputs("topic"),

        dspy.Example(
            topic="Explain quantum computing",
            key_aspects=[
                "qubits",
                "superposition",
                "entanglement",
                "quantum gates",
                "applications",
                "challenges"
            ]
        ).with_inputs("topic"),

        dspy.Example(
            topic="What is machine learning?",
            key_aspects=[
                "algorithms",
                "training data",
                "supervised learning",
                "unsupervised learning",
                "neural networks",
                "applications"
            ]
        ).with_inputs("topic"),

        dspy.Example(
            topic="Explain blockchain technology",
            key_aspects=[
                "distributed ledger",
                "cryptography",
                "consensus mechanisms",
                "smart contracts",
                "use cases",
                "limitations"
            ]
        ).with_inputs("topic"),

        dspy.Example(
            topic="What is natural language processing?",
            key_aspects=[
                "text analysis",
                "machine learning",
                "transformers",
                "tokenization",
                "applications",
                "challenges"
            ]
        ).with_inputs("topic"),

        dspy.Example(
            topic="Explain climate change",
            key_aspects=[
                "greenhouse gases",
                "global temperature",
                "human activities",
                "impacts",
                "mitigation strategies"
            ]
        ).with_inputs("topic"),

        dspy.Example(
            topic="What is artificial intelligence?",
            key_aspects=[
                "machine learning",
                "reasoning",
                "problem solving",
                "history",
                "current applications",
                "future potential"
            ]
        ).with_inputs("topic"),

        dspy.Example(
            topic="Explain the theory of evolution",
            key_aspects=[
                "natural selection",
                "genetic variation",
                "adaptation",
                "Darwin",
                "evidence",
                "modern synthesis"
            ]
        ).with_inputs("topic"),

        dspy.Example(
            topic="What is game theory?",
            key_aspects=[
                "strategic decision making",
                "Nash equilibrium",
                "prisoner's dilemma",
                "applications",
                "cooperation vs competition"
            ]
        ).with_inputs("topic"),

        dspy.Example(
            topic="Explain photosynthesis",
            key_aspects=[
                "light energy",
                "carbon dioxide",
                "glucose production",
                "chlorophyll",
                "light and dark reactions",
                "importance"
            ]
        ).with_inputs("topic"),
    ]

    return examples


# Quick test
if __name__ == "__main__":
    print("=" * 80)
    print("Research Training Examples")
    print("=" * 80)
    print()

    examples = get_training_examples()

    print(f"Total examples: {len(examples)}")
    print()

    for i, example in enumerate(examples, 1):
        print(f"{i}. {example.topic}")
        print(f"   Expected aspects ({len(example.key_aspects)}): {', '.join(example.key_aspects[:3])}...")
        print()

    print("=" * 80)
    print("✓ Training examples loaded")
    print("=" * 80)
