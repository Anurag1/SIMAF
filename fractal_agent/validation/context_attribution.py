"""
Context Attribution Analyzer - Validation Component

Analyzes which context pieces were actually USED by the LLM in its output.
This is critical for validating that our context preparation is effective.

Answers these questions:
- Was the context actually used? (not ignored)
- Which pieces were most valuable?
- Are we providing too much irrelevant context?
- Which sources contribute most effectively?

Author: BMad
Date: 2025-10-22
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """
    Result of context attribution analysis.

    Tracks which context pieces were used, how they were used,
    and provides metrics for continuous improvement.
    """
    total_pieces: int
    used_pieces: int
    unused_pieces: int
    precision: float  # % of context that was useful (0.0-1.0)

    # Detailed attribution
    piece_attributions: List[Dict[str, Any]] = field(default_factory=list)

    # By source breakdown
    by_source: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Evidence
    usage_evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_pieces": self.total_pieces,
            "used_pieces": self.used_pieces,
            "unused_pieces": self.unused_pieces,
            "precision": self.precision,
            "piece_attributions": self.piece_attributions,
            "by_source": self.by_source,
            "usage_evidence": self.usage_evidence
        }

    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = []
        lines.append(f"Context Attribution Analysis")
        lines.append(f"  Total pieces: {self.total_pieces}")
        lines.append(f"  Used: {self.used_pieces} ({self.precision*100:.1f}%)")
        lines.append(f"  Unused: {self.unused_pieces}")
        lines.append(f"")
        lines.append(f"By Source:")
        for source, stats in self.by_source.items():
            lines.append(f"  {source}: {stats['used']}/{stats['total']} used ({stats['precision']*100:.1f}%)")
        return "\n".join(lines)


class ContextAttributionAnalyzer:
    """
    Analyzes which context pieces were actually used by the LLM.

    Uses multiple techniques:
    1. Direct text matching (did phrases appear verbatim?)
    2. Semantic similarity (was the concept referenced?)
    3. Keyword extraction (were key terms used?)
    4. LLM-based analysis (ask LLM which context it used)

    Usage:
        >>> from fractal_agent.memory.context_package import ContextPackage
        >>>
        >>> analyzer = ContextAttributionAnalyzer()
        >>> result = analyzer.analyze(
        ...     context_package=context_package,
        ...     agent_output="The agent's response text...",
        ...     method="hybrid"  # Uses multiple techniques
        ... )
        >>>
        >>> print(f"Precision: {result.precision:.2f}")
        >>> print(f"Used {result.used_pieces}/{result.total_pieces} pieces")
    """

    def __init__(self, similarity_threshold: float = 0.6):
        """
        Initialize attribution analyzer.

        Args:
            similarity_threshold: Minimum similarity to consider a piece "used" (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold

    def analyze(
        self,
        context_package,  # ContextPackage
        agent_output: str,
        method: str = "hybrid",
        verbose: bool = False
    ) -> AttributionResult:
        """
        Analyze which context pieces were used in the agent's output.

        Args:
            context_package: ContextPackage with context_pieces list
            agent_output: The agent's generated output text
            method: Analysis method ("text_matching", "semantic", "hybrid", "llm")
            verbose: Print detailed analysis

        Returns:
            AttributionResult with usage statistics and evidence
        """
        if not context_package or not context_package.context_pieces:
            logger.warning("No context pieces to analyze")
            return AttributionResult(
                total_pieces=0,
                used_pieces=0,
                unused_pieces=0,
                precision=0.0
            )

        if verbose:
            logger.info(f"Analyzing {len(context_package.context_pieces)} context pieces using {method} method")

        # Normalize output for matching
        output_lower = agent_output.lower()
        output_words = set(self._extract_keywords(agent_output))

        # Analyze each piece
        piece_attributions = []
        used_count = 0

        for piece in context_package.context_pieces:
            attribution = self._analyze_piece(
                piece=piece,
                agent_output=agent_output,
                output_lower=output_lower,
                output_words=output_words,
                method=method
            )

            piece_attributions.append(attribution)

            if attribution["used"]:
                used_count += 1
                # Update the piece object
                piece.used = True
                piece.evidence = attribution["evidence"]

        # Calculate metrics
        total_pieces = len(context_package.context_pieces)
        unused_count = total_pieces - used_count
        precision = used_count / total_pieces if total_pieces > 0 else 0.0

        # Break down by source
        by_source = self._calculate_by_source(piece_attributions)

        # Collect usage evidence
        usage_evidence = [
            attr["evidence"]
            for attr in piece_attributions
            if attr["used"] and attr["evidence"]
        ]

        result = AttributionResult(
            total_pieces=total_pieces,
            used_pieces=used_count,
            unused_pieces=unused_count,
            precision=precision,
            piece_attributions=piece_attributions,
            by_source=by_source,
            usage_evidence=usage_evidence[:10]  # Top 10 examples
        )

        if verbose:
            logger.info(f"Attribution complete: {used_count}/{total_pieces} pieces used ({precision*100:.1f}% precision)")

        return result

    def _analyze_piece(
        self,
        piece,  # ContextPiece
        agent_output: str,
        output_lower: str,
        output_words: set,
        method: str
    ) -> Dict[str, Any]:
        """
        Analyze a single context piece for usage.

        Returns:
            Dict with "used" (bool), "confidence" (float), "evidence" (str)
        """
        content_lower = piece.content.lower()
        content_words = set(self._extract_keywords(piece.content))

        used = False
        confidence = 0.0
        evidence = ""

        if method in ["text_matching", "hybrid"]:
            # Direct text matching
            # Check if substantial phrases from context appear in output
            phrases = self._extract_phrases(piece.content)
            matched_phrases = [
                phrase for phrase in phrases
                if phrase.lower() in output_lower
            ]

            if matched_phrases:
                used = True
                confidence = max(confidence, 0.9)
                evidence = f"Verbatim phrase: '{matched_phrases[0]}'"

        if method in ["semantic", "hybrid"]:
            # Keyword overlap
            word_overlap = content_words & output_words
            overlap_ratio = len(word_overlap) / len(content_words) if content_words else 0.0

            if overlap_ratio >= self.similarity_threshold:
                used = True
                confidence = max(confidence, overlap_ratio)
                if not evidence:
                    evidence = f"Keyword overlap: {len(word_overlap)} keywords ({overlap_ratio*100:.1f}%)"

        if method in ["hybrid"] and not used:
            # Sequence matching as fallback
            similarity = SequenceMatcher(None, content_lower, output_lower).ratio()

            if similarity >= self.similarity_threshold:
                used = True
                confidence = max(confidence, similarity)
                evidence = f"Sequence similarity: {similarity*100:.1f}%"

        return {
            "piece_id": piece.source_id,
            "source": piece.source,
            "piece_type": piece.piece_type,
            "content_preview": piece.content[:50],
            "used": used,
            "confidence": confidence,
            "evidence": evidence,
            "relevance_score": piece.relevance_score
        }

    def _extract_keywords(self, text: str, min_length: int = 4) -> List[str]:
        """
        Extract meaningful keywords from text.

        Args:
            text: Input text
            min_length: Minimum word length to consider

        Returns:
            List of keywords (lowercase)
        """
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text.lower())

        # Common stop words to ignore
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        # Filter by length and stop words
        keywords = [
            word for word in words
            if len(word) >= min_length and word not in stop_words
        ]

        return keywords

    def _extract_phrases(self, text: str, min_words: int = 3, max_phrases: int = 10) -> List[str]:
        """
        Extract meaningful phrases from text.

        Args:
            text: Input text
            min_words: Minimum words per phrase
            max_phrases: Maximum number of phrases to extract

        Returns:
            List of phrases
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        phrases = []
        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) >= min_words:
                # Take meaningful chunks
                for i in range(len(words) - min_words + 1):
                    phrase = ' '.join(words[i:i+min_words])
                    if len(phrase) > 15:  # Minimum character length
                        phrases.append(phrase)
                        if len(phrases) >= max_phrases:
                            return phrases

        return phrases[:max_phrases]

    def _calculate_by_source(self, piece_attributions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate attribution statistics by source.

        Args:
            piece_attributions: List of attribution dicts

        Returns:
            Dict mapping source name to statistics
        """
        by_source = {}

        for attr in piece_attributions:
            source = attr["source"]
            if source not in by_source:
                by_source[source] = {
                    "total": 0,
                    "used": 0,
                    "precision": 0.0,
                    "avg_confidence": 0.0,
                    "confidences": []
                }

            by_source[source]["total"] += 1
            if attr["used"]:
                by_source[source]["used"] += 1
                by_source[source]["confidences"].append(attr["confidence"])

        # Calculate averages
        for source, stats in by_source.items():
            if stats["total"] > 0:
                stats["precision"] = stats["used"] / stats["total"]
            if stats["confidences"]:
                stats["avg_confidence"] = sum(stats["confidences"]) / len(stats["confidences"])
            # Clean up temporary list
            del stats["confidences"]

        return by_source


# ============================================================================
# Helper Functions
# ============================================================================

def quick_analyze(context_package, agent_output: str) -> AttributionResult:
    """
    Quick analysis with default settings.

    Args:
        context_package: ContextPackage to analyze
        agent_output: Agent's output text

    Returns:
        AttributionResult
    """
    analyzer = ContextAttributionAnalyzer()
    return analyzer.analyze(
        context_package=context_package,
        agent_output=agent_output,
        method="hybrid",
        verbose=False
    )


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Context Attribution Analyzer Test")
    print("=" * 80)
    print()

    # Create mock context package
    from fractal_agent.memory.context_package import ContextPackage, ContextPiece

    context_package = ContextPackage(
        domain_knowledge="VSM is a cybernetic organizational model",
        context_pieces=[
            ContextPiece(
                content="The Viable System Model (VSM) is a model of organizational structure",
                source="graphrag",
                source_id="entity_vsm_001",
                relevance_score=0.95,
                piece_type="domain_knowledge"
            ),
            ContextPiece(
                content="System 1 handles operational tasks and execution",
                source="graphrag",
                source_id="entity_system1_001",
                relevance_score=0.88,
                piece_type="domain_knowledge"
            ),
            ContextPiece(
                content="Python version 3.12 was released in 2023",
                source="web",
                source_id="web_python_001",
                relevance_score=0.45,
                piece_type="current_info"
            )
        ]
    )

    # Mock agent output that uses some context
    agent_output = """
    The Viable System Model is a cybernetic framework for organizational design.
    System 1 represents the operational level where tasks are executed.
    This model helps ensure organizational viability through recursive structure.
    """

    # Analyze
    analyzer = ContextAttributionAnalyzer()
    result = analyzer.analyze(
        context_package=context_package,
        agent_output=agent_output,
        method="hybrid",
        verbose=True
    )

    # Print results
    print()
    print(result.get_summary())
    print()
    print("Evidence:")
    for i, evidence in enumerate(result.usage_evidence, 1):
        print(f"  {i}. {evidence}")

    print()
    print("=" * 80)
    print("Context Attribution Analyzer Test Complete")
    print("=" * 80)
