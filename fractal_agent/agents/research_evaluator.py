"""
ResearchAgent Quality Evaluation for DSPy Optimization

Provides DSPy-compatible metric function for evaluating research quality.
Enables MIPRO to optimize prompts and demonstrations based on output quality.

Author: BMad
Date: 2025-10-18
"""

import dspy
from typing import Optional
from ..utils.dspy_integration import configure_dspy_balanced
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DSPy Evaluation Signatures
# ============================================================================

class EvaluateResearch(dspy.Signature):
    """
    Evaluate the quality of research synthesis.

    Check if the synthesis adequately addresses the topic with
    good coverage, accuracy, and depth.
    """
    topic = dspy.InputField(desc="The original research topic")
    synthesis = dspy.InputField(desc="The research synthesis to evaluate")
    score = dspy.OutputField(desc="Quality score 0.0-1.0 where 1.0 is excellent")
    reasoning = dspy.OutputField(desc="Brief explanation of the score")


# ============================================================================
# DSPy Metric Function
# ============================================================================

def research_quality_metric(example, pred, trace=None):
    """
    DSPy-compatible metric function for research quality evaluation.

    This is the metric function used by MIPRO to optimize the ResearchAgent.

    Args:
        example: dspy.Example with expected topic and key_aspects
        pred: ResearchResult from ResearchAgent.forward()
        trace: Optional trace information (not used)

    Returns:
        Float score 0.0-1.0 where 1.0 is perfect

    Example:
        >>> example = dspy.Example(
        ...     topic="What is VSM?",
        ...     key_aspects=["history", "5 systems", "applications"]
        ... ).with_inputs("topic")
        >>> result = agent(topic=example.topic)
        >>> score = research_quality_metric(example, result)
    """
    # Configure LLM-as-judge (use balanced tier for evaluation)
    configure_dspy_balanced()

    # Create evaluator
    evaluator = dspy.ChainOfThought(EvaluateResearch)

    try:
        # Evaluate the synthesis
        eval_result = evaluator(
            topic=example.topic,
            synthesis=pred.synthesis
        )

        # Parse score from output
        score = _parse_score(eval_result.score)

        logger.info(
            f"Evaluated '{example.topic[:50]}...' → score={score:.2f}"
        )

        return score

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 0.5  # Default middle score on error


def _parse_score(score_text: str) -> float:
    """
    Parse score from LLM output.

    Handles various formats: "0.85", "85%", "8.5/10", etc.
    """
    import re

    score_text = score_text.strip()

    # Try decimal (0.0-1.0)
    match = re.search(r'0\.\d+', score_text)
    if match:
        return min(max(float(match.group()), 0.0), 1.0)

    # Try percentage (0-100%)
    match = re.search(r'(\d+)%', score_text)
    if match:
        return min(max(float(match.group(1)) / 100.0, 0.0), 1.0)

    # Try fraction (X/10)
    match = re.search(r'(\d+\.?\d*)/(\d+)', score_text)
    if match:
        num = float(match.group(1))
        denom = float(match.group(2))
        return min(max(num / denom, 0.0), 1.0)

    # Try any number
    match = re.search(r'(\d+\.?\d*)', score_text)
    if match:
        num = float(match.group(1))
        if num <= 1.0:
            return num
        elif num <= 10.0:
            return num / 10.0
        elif num <= 100.0:
            return num / 100.0

    logger.warning(f"Could not parse score from: {score_text}")
    return 0.5  # Default


# ============================================================================
# Simple Heuristic Metric (No LLM)
# ============================================================================

def research_completeness_metric(example, pred, trace=None):
    """
    Simple heuristic metric without LLM evaluation.

    Checks if synthesis mentions expected key aspects.
    Faster but less nuanced than LLM-as-judge.

    Args:
        example: dspy.Example with topic and key_aspects list
        pred: ResearchResult from agent
        trace: Not used

    Returns:
        Float 0.0-1.0 based on aspect coverage
    """
    if not hasattr(example, 'key_aspects'):
        return 1.0  # No aspects to check

    synthesis_lower = pred.synthesis.lower()
    aspects_found = sum(
        1 for aspect in example.key_aspects
        if aspect.lower() in synthesis_lower
    )

    score = aspects_found / len(example.key_aspects)
    return score


# Quick test
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    from .research_agent import ResearchAgent

    print("=" * 80)
    print("ResearchEvaluator Test")
    print("=" * 80)
    print()

    # Create example
    example = dspy.Example(
        topic="What is 2+2?",
        key_aspects=["arithmetic", "addition", "equals 4"]
    ).with_inputs("topic")

    print(f"Example topic: {example.topic}")
    print(f"Expected aspects: {example.key_aspects}")
    print()

    # Run research
    print("Running research...")
    agent = ResearchAgent(max_research_questions=1)
    result = agent(topic=example.topic, verbose=False)
    print(f"✓ Research complete")
    print(f"  Synthesis: {result.synthesis[:200]}...")
    print()

    # Test heuristic metric
    print("Test 1: Heuristic metric (fast, no LLM)")
    print("-" * 80)
    score1 = research_completeness_metric(example, result)
    print(f"Score: {score1:.2f}")
    print()

    # Test LLM-as-judge metric
    print("Test 2: LLM-as-judge metric (slower, more nuanced)")
    print("-" * 80)
    score2 = research_quality_metric(example, result)
    print(f"Score: {score2:.2f}")
    print()

    print("=" * 80)
    print("✓ ResearchEvaluator test complete!")
    print("=" * 80)
