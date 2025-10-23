"""
MIPRO Optimization for ResearchAgent

Uses DSPy's MIPRO optimizer to find the best prompts and demonstrations
for the ResearchAgent based on quality metrics.

Author: BMad
Date: 2025-10-18
"""

import dspy
from dspy.teleprompt import MIPROv2
from .research_agent import ResearchAgent
from .research_config import ResearchConfig
from .research_evaluator import research_quality_metric, research_completeness_metric
from .research_examples import get_training_examples
from ..utils.dspy_integration import configure_dspy_balanced
import logging

logger = logging.getLogger(__name__)


def optimize_research_agent(
    config: ResearchConfig = None,
    use_llm_judge: bool = True,
    auto: str = "light",  # light/medium/heavy - controls optimization depth
    num_candidates: int = None,  # Only used if auto=None
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 4,
    save_path: str = "optimized_research_agent.json"
):
    """
    Optimize ResearchAgent using MIPRO.

    This runs DSPy's MIPRO optimizer to find the best:
    - Instructions for each signature (ResearchPlanning, InformationGathering, etc.)
    - Few-shot demonstrations for each stage
    - Prompt structure and wording

    Args:
        config: ResearchConfig to use (default: default config)
        use_llm_judge: Use LLM-as-judge metric (slower but better) vs heuristic
        auto: Auto mode (light/medium/heavy) or None for manual control
        num_candidates: Number of candidate programs (only if auto=None)
        max_bootstrapped_demos: Max few-shot examples to bootstrap
        max_labeled_demos: Max labeled examples to use
        save_path: Where to save optimized agent

    Returns:
        Optimized ResearchAgent

    Example:
        >>> optimized_agent = optimize_research_agent()
        >>> result = optimized_agent(topic="Your topic")
    """
    print("=" * 80)
    print("MIPRO Optimization for ResearchAgent")
    print("=" * 80)
    print()

    # Load training examples
    print("Loading training examples...")
    trainset = get_training_examples()
    print(f"✓ Loaded {len(trainset)} training examples")
    print()

    # Configure DSPy
    print("Configuring DSPy...")
    configure_dspy_balanced()
    print("✓ DSPy configured")
    print()

    # Create agent
    print("Creating ResearchAgent...")
    config = config if config is not None else ResearchConfig()
    agent = ResearchAgent(config=config, max_research_questions=2)  # Limit for speed
    print(f"✓ Agent created with config: {config}")
    print()

    # Select metric
    metric = research_quality_metric if use_llm_judge else research_completeness_metric
    metric_name = "LLM-as-judge" if use_llm_judge else "Heuristic"
    print(f"Using metric: {metric_name}")
    print()

    # Initialize MIPRO
    print("Initializing MIPRO optimizer...")
    if auto:
        print(f"  auto={auto} (will auto-determine num_candidates)")
        optimizer = MIPROv2(
            metric=metric,
            auto=auto
        )
    else:
        print(f"  num_candidates={num_candidates}")
        optimizer = MIPROv2(
            metric=metric,
            num_candidates=num_candidates
        )

    print(f"  max_bootstrapped_demos={max_bootstrapped_demos}")
    print(f"  max_labeled_demos={max_labeled_demos}")
    print()

    print("✓ MIPRO initialized")
    print()

    # Run optimization
    print("=" * 80)
    print("STARTING OPTIMIZATION (this may take several minutes)")
    print("=" * 80)
    print()

    try:
        optimized_agent = optimizer.compile(
            agent,
            trainset=trainset,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            requires_permission_to_run=False  # Auto-run without prompts
        )

        print()
        print("=" * 80)
        print("✓ OPTIMIZATION COMPLETE")
        print("=" * 80)
        print()

        # Save optimized agent
        print(f"Saving optimized agent to {save_path}...")
        optimized_agent.save(save_path)
        print(f"✓ Saved to {save_path}")
        print()

        return optimized_agent

    except Exception as e:
        print(f"\n✗ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# Quick test
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("This script runs MIPRO optimization on ResearchAgent.")
    print("WARNING: This will take several minutes and use significant tokens.")
    print()

    # Ask for confirmation
    response = input("Do you want to proceed? (yes/no): ").strip().lower()

    if response == "yes":
        print()
        optimized_agent = optimize_research_agent(
            use_llm_judge=False,  # Use heuristic for faster testing
            num_candidates=5,  # Fewer candidates for testing
            max_bootstrapped_demos=2,
            max_labeled_demos=3
        )

        if optimized_agent:
            print()
            print("Testing optimized agent...")
            result = optimized_agent(
                topic="What is the Viable System Model?",
                verbose=False
            )
            print(f"✓ Test successful!")
            print(f"  Synthesis: {result.synthesis[:200]}...")
    else:
        print("Optimization cancelled.")
