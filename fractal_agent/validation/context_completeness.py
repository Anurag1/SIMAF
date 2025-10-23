"""
Context Completeness Evaluator - Validation Component

Evaluates whether the provided context was COMPLETE and SUFFICIENT.
This answers: "Did the agent have everything it needed?"

While attribution checks if context was USED (precision),
completeness checks if context was ENOUGH (recall).

Indicators of Incomplete Context:
- Agent makes unsupported assumptions
- Output contains hedging language ("might", "possibly", "uncertain")
- Agent explicitly states missing information
- Key concepts undefined or poorly explained
- Errors or hallucinations due to knowledge gaps

Author: BMad
Date: 2025-10-22
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CompletenessResult:
    """
    Result of context completeness evaluation.

    Tracks whether the agent had sufficient context,
    what was missing, and how to improve.
    """
    completeness_score: float  # 0.0 (very incomplete) to 1.0 (complete)
    is_complete: bool  # True if score >= threshold

    # Gap analysis
    identified_gaps: List[str] = field(default_factory=list)
    missing_concepts: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)

    # Indicators
    hedging_count: int = 0  # "might", "possibly", "uncertain"
    assumption_count: int = 0  # "assuming", "likely", "probably"
    error_indicators: List[str] = field(default_factory=list)

    # Recommendations
    suggested_additions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "completeness_score": self.completeness_score,
            "is_complete": self.is_complete,
            "identified_gaps": self.identified_gaps,
            "missing_concepts": self.missing_concepts,
            "unsupported_claims": self.unsupported_claims,
            "hedging_count": self.hedging_count,
            "assumption_count": self.assumption_count,
            "error_indicators": self.error_indicators,
            "suggested_additions": self.suggested_additions
        }

    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = []
        lines.append(f"Context Completeness Evaluation")
        lines.append(f"  Completeness: {self.completeness_score*100:.1f}%")
        lines.append(f"  Status: {'✓ Complete' if self.is_complete else '✗ Incomplete'}")
        lines.append(f"  Hedging: {self.hedging_count} instances")
        lines.append(f"  Assumptions: {self.assumption_count} instances")
        lines.append(f"")

        if self.identified_gaps:
            lines.append(f"Identified Gaps:")
            for gap in self.identified_gaps[:5]:
                lines.append(f"  • {gap}")

        if self.suggested_additions:
            lines.append(f"")
            lines.append(f"Suggested Additions:")
            for suggestion in self.suggested_additions[:3]:
                lines.append(f"  • {suggestion}")

        return "\n".join(lines)


class CompletenessEvaluator:
    """
    Evaluates whether context was complete and sufficient.

    Uses multiple heuristics:
    1. Hedging language detection ("might", "possibly", "unclear")
    2. Assumption detection ("assuming", "likely", "probably")
    3. Explicit gap statements ("I don't know", "unclear", "missing")
    4. Error pattern detection (hallucinations, contradictions)
    5. Concept coverage analysis

    Usage:
        >>> evaluator = CompletenessEvaluator()
        >>> result = evaluator.evaluate(
        ...     context_package=context_package,
        ...     agent_output="The agent's response...",
        ...     user_task="Original task..."
        ... )
        >>>
        >>> print(f"Completeness: {result.completeness_score:.2f}")
        >>> if not result.is_complete:
        ...     print(f"Missing: {result.identified_gaps}")
    """

    def __init__(
        self,
        completeness_threshold: float = 0.8,
        hedging_weight: float = 0.3,
        assumption_weight: float = 0.4,
        gap_weight: float = 0.8
    ):
        """
        Initialize completeness evaluator.

        Args:
            completeness_threshold: Minimum score to be considered "complete"
            hedging_weight: How much hedging reduces completeness (0.0-1.0)
            assumption_weight: How much assumptions reduce completeness (0.0-1.0)
            gap_weight: How much explicit gaps reduce completeness (0.0-1.0)
        """
        self.completeness_threshold = completeness_threshold
        self.hedging_weight = hedging_weight
        self.assumption_weight = assumption_weight
        self.gap_weight = gap_weight

        # Hedging language patterns
        self.hedging_patterns = [
            r'\b(might|maybe|possibly|perhaps|could be|uncertain|unclear)\b',
            r'\b(not sure|don\'t know|unsure|questionable)\b',
            r'\b(seems to|appears to|tends to)\b'
        ]

        # Assumption patterns
        self.assumption_patterns = [
            r'\b(assuming|assuming that|assume|likely|probably|presumably)\b',
            r'\b(if we assume|given that|provided that)\b',
            r'\b(it\'s likely|it seems likely)\b'
        ]

        # Explicit gap patterns
        self.gap_patterns = [
            r'\b(I don\'t (know|have)|unclear|missing information|need more)\b',
            r'\b(insufficient (data|information|context)|lacks (detail|information))\b',
            r'\b(would need to|requires additional|more information needed)\b'
        ]

    def evaluate(
        self,
        context_package,  # ContextPackage
        agent_output: str,
        user_task: str,
        verbose: bool = False
    ) -> CompletenessResult:
        """
        Evaluate completeness of context for the given task.

        Args:
            context_package: ContextPackage that was provided
            agent_output: Agent's generated output
            user_task: Original user task/request
            verbose: Print detailed analysis

        Returns:
            CompletenessResult with completeness score and gap analysis
        """
        if verbose:
            logger.info(f"Evaluating context completeness for task: {user_task[:50]}...")

        # Count indicators
        hedging_count, hedging_examples = self._count_patterns(
            agent_output,
            self.hedging_patterns,
            "hedging"
        )

        assumption_count, assumption_examples = self._count_patterns(
            agent_output,
            self.assumption_patterns,
            "assumptions"
        )

        gap_count, gap_examples = self._count_patterns(
            agent_output,
            self.gap_patterns,
            "gaps"
        )

        # Identify what's missing
        identified_gaps = gap_examples
        missing_concepts = self._identify_missing_concepts(
            user_task,
            context_package,
            agent_output
        )

        unsupported_claims = self._find_unsupported_claims(
            agent_output,
            context_package
        )

        # Calculate completeness score
        # Start at 100%, reduce based on indicators
        score = 1.0

        # Reduce for hedging (each hedge reduces slightly)
        hedge_penalty = min(hedging_count * 0.05 * self.hedging_weight, 0.3)
        score -= hedge_penalty

        # Reduce for assumptions (each assumption reduces more)
        assumption_penalty = min(assumption_count * 0.08 * self.assumption_weight, 0.4)
        score -= assumption_penalty

        # Reduce for explicit gaps (each gap reduces significantly)
        gap_penalty = min(gap_count * 0.15 * self.gap_weight, 0.5)
        score -= gap_penalty

        # Reduce for unsupported claims
        unsupported_penalty = min(len(unsupported_claims) * 0.1, 0.2)
        score -= unsupported_penalty

        # Clamp to [0.0, 1.0]
        score = max(0.0, min(1.0, score))

        # Generate suggestions
        suggested_additions = self._generate_suggestions(
            missing_concepts,
            unsupported_claims,
            gap_examples
        )

        result = CompletenessResult(
            completeness_score=score,
            is_complete=score >= self.completeness_threshold,
            identified_gaps=identified_gaps,
            missing_concepts=missing_concepts,
            unsupported_claims=unsupported_claims,
            hedging_count=hedging_count,
            assumption_count=assumption_count,
            error_indicators=gap_examples,
            suggested_additions=suggested_additions
        )

        if verbose:
            logger.info(f"Completeness: {score*100:.1f}% ({'complete' if result.is_complete else 'incomplete'})")
            logger.info(f"  Hedging: {hedging_count}, Assumptions: {assumption_count}, Gaps: {gap_count}")

        return result

    def _count_patterns(
        self,
        text: str,
        patterns: List[str],
        label: str
    ) -> tuple:
        """
        Count occurrences of patterns in text.

        Args:
            text: Text to search
            patterns: List of regex patterns
            label: Label for logging

        Returns:
            (count, examples) tuple
        """
        examples = []
        count = 0

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                count += 1
                # Get context around match
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                examples.append(context)
                if len(examples) >= 5:  # Max 5 examples
                    break

        return count, examples

    def _identify_missing_concepts(
        self,
        user_task: str,
        context_package,
        agent_output: str
    ) -> List[str]:
        """
        Identify concepts mentioned in task but missing from context.

        Args:
            user_task: Original user task
            context_package: Provided context
            agent_output: Agent's output

        Returns:
            List of potentially missing concepts
        """
        missing = []

        # Extract key terms from task
        task_terms = self._extract_key_terms(user_task)

        # Check if context covers these terms
        if context_package:
            context_text = (
                context_package.domain_knowledge + " " +
                context_package.constraints + " " +
                context_package.current_info
            ).lower()

            for term in task_terms:
                if term.lower() not in context_text:
                    # Check if it appears in output (might be hallucinated)
                    if term.lower() in agent_output.lower():
                        missing.append(f"'{term}' (in output but not in context)")

        return missing[:5]  # Top 5

    def _find_unsupported_claims(
        self,
        agent_output: str,
        context_package
    ) -> List[str]:
        """
        Find claims in output that aren't supported by context.

        This is a simplified heuristic - looks for definitive statements
        that don't appear in the context.

        Args:
            agent_output: Agent's output
            context_package: Provided context

        Returns:
            List of potentially unsupported claims
        """
        unsupported = []

        if not context_package:
            return unsupported

        # Get all context text
        context_text = (
            context_package.domain_knowledge + " " +
            context_package.constraints + " " +
            context_package.current_info
        ).lower()

        # Extract definitive statements from output
        # Look for sentences with strong verbs: "is", "are", "will", "must"
        sentences = re.split(r'[.!?]+', agent_output)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short
                continue

            # Check if it's a definitive statement
            if re.search(r'\b(is|are|will|must|should|cannot)\b', sentence, re.IGNORECASE):
                # Check if key parts appear in context
                sentence_lower = sentence.lower()
                words = [w for w in sentence_lower.split() if len(w) > 4]

                if words:
                    # If less than 30% of significant words appear in context, it might be unsupported
                    matched = sum(1 for w in words if w in context_text)
                    ratio = matched / len(words)

                    if ratio < 0.3:
                        unsupported.append(sentence[:100])
                        if len(unsupported) >= 3:
                            break

        return unsupported

    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from text.

        Args:
            text: Input text

        Returns:
            List of key terms
        """
        # Simple extraction: words 5+ chars, capitalized or technical
        words = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]{5,}\b', text)

        # Deduplicate and limit
        unique_words = list(dict.fromkeys(words))
        return unique_words[:10]

    def _generate_suggestions(
        self,
        missing_concepts: List[str],
        unsupported_claims: List[str],
        gaps: List[str]
    ) -> List[str]:
        """
        Generate suggestions for improving context.

        Args:
            missing_concepts: Concepts that were missing
            unsupported_claims: Claims without support
            gaps: Explicit gaps found

        Returns:
            List of actionable suggestions
        """
        suggestions = []

        if missing_concepts:
            suggestions.append(
                f"Add domain knowledge for: {', '.join(missing_concepts[:3])}"
            )

        if unsupported_claims:
            suggestions.append(
                "Add supporting evidence for claims made in output"
            )

        if gaps:
            suggestions.append(
                "Address explicit information gaps mentioned in output"
            )

        # Generic suggestions based on patterns
        if len(suggestions) == 0:
            suggestions.append("Context appears mostly complete")

        return suggestions[:5]


# ============================================================================
# Helper Functions
# ============================================================================

def quick_evaluate(
    context_package,
    agent_output: str,
    user_task: str
) -> CompletenessResult:
    """
    Quick evaluation with default settings.

    Args:
        context_package: ContextPackage to evaluate
        agent_output: Agent's output text
        user_task: Original user task

    Returns:
        CompletenessResult
    """
    evaluator = CompletenessEvaluator()
    return evaluator.evaluate(
        context_package=context_package,
        agent_output=agent_output,
        user_task=user_task,
        verbose=False
    )


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Context Completeness Evaluator Test")
    print("=" * 80)
    print()

    # Create mock context
    from fractal_agent.memory.context_package import ContextPackage, ContextPiece

    context_package = ContextPackage(
        domain_knowledge="VSM is a cybernetic organizational model with 5 systems.",
        constraints="Follow system theory principles",
        current_info=""
    )

    # Test Case 1: Complete output (no hedging, no gaps)
    complete_output = """
    The Viable System Model consists of five interconnected systems that ensure
    organizational viability. System 1 handles operations, System 2 coordinates,
    System 3 optimizes, System 4 handles intelligence, and System 5 sets policy.
    """

    # Test Case 2: Incomplete output (hedging and gaps)
    incomplete_output = """
    The Viable System Model might have five systems, though I'm not entirely sure.
    System 1 probably handles operations, assuming that's the operational level.
    I don't have enough information about System 4's specific role, but it seems
    to be related to intelligence gathering. More information needed on System 5.
    """

    evaluator = CompletenessEvaluator()

    print("Test 1: Complete Output")
    print("-" * 80)
    result1 = evaluator.evaluate(
        context_package=context_package,
        agent_output=complete_output,
        user_task="Explain the Viable System Model",
        verbose=True
    )
    print(result1.get_summary())

    print()
    print("=" * 80)
    print()

    print("Test 2: Incomplete Output")
    print("-" * 80)
    result2 = evaluator.evaluate(
        context_package=context_package,
        agent_output=incomplete_output,
        user_task="Explain the Viable System Model",
        verbose=True
    )
    print(result2.get_summary())

    print()
    print("=" * 80)
    print("Context Completeness Evaluator Test Complete")
    print("=" * 80)
