"""
Knowledge Validation Framework

Provides comprehensive validation capabilities including:
- Fact-checking against trusted sources
- Authority scoring for information sources
- Conflict detection between knowledge items
- Currency tracking for information freshness
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
from collections import defaultdict


class ValidationStatus(Enum):
    """Status of validation check."""
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"
    OUTDATED = "outdated"
    PENDING = "pending"


class AuthorityLevel(Enum):
    """Authority level of information source."""
    PRIMARY = 5  # Original research, official documentation
    SECONDARY = 4  # Peer-reviewed analysis, expert commentary
    TERTIARY = 3  # Textbooks, encyclopedias
    INFORMAL = 2  # Blogs, forums with credentials
    UNVERIFIED = 1  # Unknown or unverified sources


@dataclass
class Source:
    """Represents an information source."""
    identifier: str
    name: str
    url: Optional[str] = None
    authority_level: AuthorityLevel = AuthorityLevel.UNVERIFIED
    domain: Optional[str] = None
    credentials: List[str] = field(default_factory=list)
    trust_score: float = 0.5
    last_verified: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.identifier)


@dataclass
class Fact:
    """Represents a factual claim."""
    content: str
    source: Source
    timestamp: datetime
    domain: str
    confidence: float = 0.5
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def fact_id(self) -> str:
        """Generate unique identifier for fact."""
        content_hash = hashlib.sha256(
            f"{self.content}{self.source.identifier}".encode()
        ).hexdigest()[:16]
        return content_hash

    @property
    def age(self) -> timedelta:
        """Get age of the fact."""
        return datetime.now() - self.timestamp


@dataclass
class ValidationResult:
    """Result of validation check."""
    fact: Fact
    status: ValidationStatus
    confidence: float
    supporting_sources: List[Source] = field(default_factory=list)
    conflicting_sources: List[Source] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    conflicts: List['Conflict'] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    notes: str = ""


@dataclass
class Conflict:
    """Represents a conflict between facts."""
    fact_a: Fact
    fact_b: Fact
    conflict_type: str
    severity: float
    resolution: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.now)


class AuthorityScorer:
    """Scores the authority and credibility of sources."""

    def __init__(self):
        self.trusted_domains: Set[str] = {
            "edu", "gov", "org", "ac.uk", "arxiv.org"
        }
        self.domain_scores: Dict[str, float] = {}
        self.source_history: Dict[str, List[float]] = defaultdict(list)

    def score_source(self, source: Source) -> float:
        """
        Calculate comprehensive authority score for a source.

        Args:
            source: Source to score

        Returns:
            Authority score between 0.0 and 1.0
        """
        score = 0.0

        # Base score from authority level
        score += source.authority_level.value / 5.0 * 0.4

        # Domain trust bonus
        if source.domain:
            domain_suffix = source.domain.split('.')[-1]
            if domain_suffix in self.trusted_domains:
                score += 0.2
            if source.domain in self.domain_scores:
                score += self.domain_scores[source.domain] * 0.2

        # Credentials bonus
        score += min(len(source.credentials) * 0.05, 0.2)

        # Existing trust score
        score += source.trust_score * 0.2

        # Historical performance
        if source.identifier in self.source_history:
            history = self.source_history[source.identifier]
            avg_history = sum(history) / len(history)
            score += avg_history * 0.1

        return min(score, 1.0)

    def update_source_score(
        self,
        source: Source,
        performance: float
    ) -> None:
        """
        Update source score based on validation performance.

        Args:
            source: Source to update
            performance: Performance score (0.0-1.0)
        """
        self.source_history[source.identifier].append(performance)

        # Keep only recent history
        if len(self.source_history[source.identifier]) > 100:
            self.source_history[source.identifier].pop(0)

        # Update trust score with exponential moving average
        alpha = 0.1
        source.trust_score = (
            alpha * performance + (1 - alpha) * source.trust_score
        )

    def add_trusted_domain(self, domain: str, score: float = 1.0) -> None:
        """Add a domain to trusted list with score."""
        self.trusted_domains.add(domain)
        self.domain_scores[domain] = score


class FactChecker:
    """Verifies facts against trusted sources."""

    def __init__(self, authority_scorer: AuthorityScorer):
        self.authority_scorer = authority_scorer
        self.verified_facts: Dict[str, ValidationResult] = {}
        self.fact_index: Dict[str, Set[Fact]] = defaultdict(set)

    def check_fact(
        self,
        fact: Fact,
        reference_facts: Optional[List[Fact]] = None
    ) -> ValidationResult:
        """
        Check a fact against reference facts and sources.

        Args:
            fact: Fact to verify
            reference_facts: Optional list of reference facts to check against

        Returns:
            ValidationResult with verification status
        """
        if reference_facts is None:
            reference_facts = []

        supporting_sources = []
        conflicting_sources = []
        evidence = []

        # Score the fact's source
        source_score = self.authority_scorer.score_source(fact.source)

        # Check against reference facts
        for ref_fact in reference_facts:
            ref_score = self.authority_scorer.score_source(ref_fact.source)
            similarity = self._calculate_similarity(fact, ref_fact)

            if similarity > 0.8:
                supporting_sources.append(ref_fact.source)
                evidence.append(f"Supported by: {ref_fact.source.name}")
            elif similarity < -0.5:
                conflicting_sources.append(ref_fact.source)
                evidence.append(f"Contradicted by: {ref_fact.source.name}")

        # Determine status
        status = self._determine_status(
            fact,
            len(supporting_sources),
            len(conflicting_sources),
            source_score
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            source_score,
            supporting_sources,
            conflicting_sources
        )

        result = ValidationResult(
            fact=fact,
            status=status,
            confidence=confidence,
            supporting_sources=supporting_sources,
            conflicting_sources=conflicting_sources,
            evidence=evidence
        )

        # Store result
        self.verified_facts[fact.fact_id] = result
        self.fact_index[fact.domain].add(fact)

        return result

    def _calculate_similarity(self, fact_a: Fact, fact_b: Fact) -> float:
        """
        Calculate semantic similarity between facts.

        Returns:
            Similarity score from -1.0 (contradictory) to 1.0 (identical)
        """
        # Simple implementation - would use embeddings in production
        content_a = fact_a.content.lower()
        content_b = fact_b.content.lower()

        # Exact match
        if content_a == content_b:
            return 1.0

        # Check for contradiction keywords
        contradiction_pairs = [
            ("true", "false"),
            ("correct", "incorrect"),
            ("valid", "invalid"),
            ("is", "is not"),
            ("does", "does not")
        ]

        for word_a, word_b in contradiction_pairs:
            if (word_a in content_a and word_b in content_b) or \
               (word_b in content_a and word_a in content_b):
                return -0.8

        # Simple word overlap
        words_a = set(content_a.split())
        words_b = set(content_b.split())

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        jaccard = len(intersection) / len(union)
        return jaccard * 0.7  # Scale down simple similarity

    def _determine_status(
        self,
        fact: Fact,
        supporting_count: int,
        conflicting_count: int,
        source_score: float
    ) -> ValidationStatus:
        """Determine validation status based on evidence."""
        # Check currency
        if fact.age > timedelta(days=365) and source_score < 0.7:
            return ValidationStatus.OUTDATED

        # Check conflicts
        if conflicting_count > supporting_count:
            return ValidationStatus.CONTRADICTED

        # Check verification
        if supporting_count >= 2 or (supporting_count >= 1 and source_score > 0.8):
            return ValidationStatus.VERIFIED

        if supporting_count == 0 and source_score > 0.6:
            return ValidationStatus.UNVERIFIED

        return ValidationStatus.PENDING

    def _calculate_confidence(
        self,
        source_score: float,
        supporting_sources: List[Source],
        conflicting_sources: List[Source]
    ) -> float:
        """Calculate overall confidence in fact."""
        confidence = source_score * 0.4

        # Add support bonus
        support_scores = [
            self.authority_scorer.score_source(s)
            for s in supporting_sources
        ]
        if support_scores:
            confidence += min(sum(support_scores) / len(support_scores), 0.4)

        # Subtract conflict penalty
        conflict_scores = [
            self.authority_scorer.score_source(s)
            for s in conflicting_sources
        ]
        if conflict_scores:
            penalty = min(sum(conflict_scores) / len(conflict_scores), 0.3)
            confidence -= penalty

        return max(0.0, min(1.0, confidence))


class ConflictDetector:
    """Detects conflicts and contradictions between knowledge items."""

    def __init__(self):
        self.conflicts: List[Conflict] = []
        self.conflict_index: Dict[str, List[Conflict]] = defaultdict(list)

    def detect_conflicts(
        self,
        facts: List[Fact]
    ) -> List[Conflict]:
        """
        Detect conflicts among a set of facts.

        Args:
            facts: List of facts to check for conflicts

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Group facts by domain
        domain_groups = defaultdict(list)
        for fact in facts:
            domain_groups[fact.domain].append(fact)

        # Check within domains
        for domain, domain_facts in domain_groups.items():
            for i, fact_a in enumerate(domain_facts):
                for fact_b in domain_facts[i+1:]:
                    conflict = self._check_pair(fact_a, fact_b)
                    if conflict:
                        conflicts.append(conflict)
                        self.conflicts.append(conflict)
                        self.conflict_index[fact_a.fact_id].append(conflict)
                        self.conflict_index[fact_b.fact_id].append(conflict)

        return conflicts

    def _check_pair(self, fact_a: Fact, fact_b: Fact) -> Optional[Conflict]:
        """Check if two facts conflict."""
        content_a = fact_a.content.lower()
        content_b = fact_b.content.lower()

        # Negation patterns
        negation_patterns = [
            ("is", "is not"),
            ("are", "are not"),
            ("does", "does not"),
            ("can", "cannot"),
            ("will", "will not"),
            ("has", "has not"),
            ("true", "false"),
            ("correct", "incorrect"),
            ("valid", "invalid")
        ]

        conflict_type = None
        severity = 0.0

        for pos, neg in negation_patterns:
            if (pos in content_a and neg in content_b) or \
               (neg in content_a and pos in content_b):
                conflict_type = "negation_conflict"
                severity = 0.9
                break

        # Numerical conflicts
        if not conflict_type:
            import re
            numbers_a = re.findall(r'\d+\.?\d*', content_a)
            numbers_b = re.findall(r'\d+\.?\d*', content_b)

            if numbers_a and numbers_b:
                try:
                    val_a = float(numbers_a[0])
                    val_b = float(numbers_b[0])
                    if abs(val_a - val_b) / max(val_a, val_b) > 0.1:
                        conflict_type = "numerical_conflict"
                        severity = min(abs(val_a - val_b) / max(val_a, val_b), 1.0)
                except ValueError:
                    pass

        # Temporal conflicts
        if not conflict_type and fact_a.timestamp and fact_b.timestamp:
            time_diff = abs((fact_a.timestamp - fact_b.timestamp).days)
            if time_diff > 365:
                # Same claim with large time gap might indicate outdated info
                words_a = set(content_a.split())
                words_b = set(content_b.split())
                overlap = len(words_a & words_b) / len(words_a | words_b)
                if overlap > 0.6:
                    conflict_type = "temporal_conflict"
                    severity = min(time_diff / 3650, 0.7)

        if conflict_type:
            return Conflict(
                fact_a=fact_a,
                fact_b=fact_b,
                conflict_type=conflict_type,
                severity=severity
            )

        return None

    def resolve_conflict(
        self,
        conflict: Conflict,
        resolution: str
    ) -> None:
        """Record resolution of a conflict."""
        conflict.resolution = resolution

    def get_conflicts_for_fact(self, fact_id: str) -> List[Conflict]:
        """Get all conflicts involving a specific fact."""
        return self.conflict_index.get(fact_id, [])


class CurrencyTracker:
    """Tracks freshness and currency of information."""

    def __init__(self):
        self.freshness_thresholds: Dict[str, timedelta] = {
            "news": timedelta(days=7),
            "technology": timedelta(days=90),
            "science": timedelta(days=365),
            "history": timedelta(days=3650),
            "general": timedelta(days=365)
        }
        self.update_log: List[Dict[str, Any]] = []

    def check_currency(self, fact: Fact) -> Tuple[bool, float]:
        """
        Check if a fact is current.

        Args:
            fact: Fact to check

        Returns:
            Tuple of (is_current, staleness_score)
        """
        threshold = self.freshness_thresholds.get(
            fact.domain,
            self.freshness_thresholds["general"]
        )

        age = fact.age

        is_current = age <= threshold

        # Calculate staleness score (0.0 = fresh, 1.0 = very stale)
        staleness = min(age / threshold, 2.0) / 2.0

        return is_current, staleness

    def set_freshness_threshold(
        self,
        domain: str,
        threshold: timedelta
    ) -> None:
        """Set freshness threshold for a domain."""
        self.freshness_thresholds[domain] = threshold

    def track_update(
        self,
        fact_id: str,
        old_fact: Fact,
        new_fact: Fact
    ) -> None:
        """Track when a fact is updated."""
        self.update_log.append({
            "fact_id": fact_id,
            "old_timestamp": old_fact.timestamp,
            "new_timestamp": new_fact.timestamp,
            "update_time": datetime.now(),
            "old_content": old_fact.content,
            "new_content": new_fact.content
        })

    def get_stale_facts(
        self,
        facts: List[Fact],
        threshold: Optional[float] = 0.5
    ) -> List[Tuple[Fact, float]]:
        """
        Get facts that are stale.

        Args:
            facts: List of facts to check
            threshold: Staleness threshold (0.0-1.0)

        Returns:
            List of (fact, staleness_score) tuples
        """
        stale_facts = []

        for fact in facts:
            is_current, staleness = self.check_currency(fact)
            if staleness >= threshold:
                stale_facts.append((fact, staleness))

        return sorted(stale_facts, key=lambda x: x[1], reverse=True)


class KnowledgeValidator:
    """
    Main knowledge validation framework.

    Integrates fact-checking, authority scoring, conflict detection,
    and currency tracking.
    """

    def __init__(self):
        self.authority_scorer = AuthorityScorer()
        self.fact_checker = FactChecker(self.authority_scorer)
        self.conflict_detector = ConflictDetector()
        self.currency_tracker = CurrencyTracker()

        self.knowledge_base: Dict[str, Fact] = {}
        self.validation_history: List[ValidationResult] = []

    def add_fact(
        self,
        fact: Fact,
        validate: bool = True
    ) -> Optional[ValidationResult]:
        """
        Add a fact to the knowledge base.

        Args:
            fact: Fact to add
            validate: Whether to validate immediately

        Returns:
            ValidationResult if validated, None otherwise
        """
        self.knowledge_base[fact.fact_id] = fact

        if validate:
            return self.validate_fact(fact)

        return None

    def validate_fact(self, fact: Fact) -> ValidationResult:
        """
        Comprehensively validate a fact.

        Args:
            fact: Fact to validate

        Returns:
            Complete validation result
        """
        # Get reference facts from same domain
        reference_facts = [
            f for f in self.knowledge_base.values()
            if f.domain == fact.domain and f.fact_id != fact.fact_id
        ]

        # Fact-check
        result = self.fact_checker.check_fact(fact, reference_facts)

        # Check currency
        is_current, staleness = self.currency_tracker.check_currency(fact)
        if not is_current:
            result.status = ValidationStatus.OUTDATED
            result.notes += f" Staleness score: {staleness:.2f}"

        # Detect conflicts
        all_facts = list(self.knowledge_base.values()) + [fact]
        conflicts = self.conflict_detector.detect_conflicts(all_facts)

        # Add conflicts to result
        fact_conflicts = [
            c for c in conflicts
            if c.fact_a.fact_id == fact.fact_id or c.fact_b.fact_id == fact.fact_id
        ]
        result.conflicts = fact_conflicts

        if fact_conflicts and result.status != ValidationStatus.CONTRADICTED:
            result.status = ValidationStatus.CONTRADICTED

        # Store validation
        self.validation_history.append(result)

        return result

    def validate_all(self) -> List[ValidationResult]:
        """Validate all facts in knowledge base."""
        results = []
        for fact in self.knowledge_base.values():
            result = self.validate_fact(fact)
            results.append(result)
        return results

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation status."""
        if not self.validation_history:
            return {"total": 0}

        status_counts = defaultdict(int)
        for result in self.validation_history:
            status_counts[result.status.value] += 1

        avg_confidence = sum(
            r.confidence for r in self.validation_history
        ) / len(self.validation_history)

        return {
            "total": len(self.validation_history),
            "by_status": dict(status_counts),
            "average_confidence": avg_confidence,
            "conflicts_detected": len(self.conflict_detector.conflicts),
            "sources_tracked": len(self.authority_scorer.source_history)
        }

    def export_validation_report(self) -> str:
        """Export detailed validation report as JSON."""
        report = {
            "summary": self.get_validation_summary(),
            "validations": [
                {
                    "fact_id": r.fact.fact_id,
                    "content": r.fact.content,
                    "status": r.status.value,
                    "confidence": r.confidence,
                    "source": r.fact.source.name,
                    "timestamp": r.timestamp.isoformat(),
                    "conflicts": len(r.conflicts)
                }
                for r in self.validation_history
            ],
            "conflicts": [
                {
                    "type": c.conflict_type,
                    "severity": c.severity,
                    "fact_a": c.fact_a.content[:100],
                    "fact_b": c.fact_b.content[:100],
                    "resolved": c.resolution is not None
                }
                for c in self.conflict_detector.conflicts
            ]
        }
        return json.dumps(report, indent=2)
