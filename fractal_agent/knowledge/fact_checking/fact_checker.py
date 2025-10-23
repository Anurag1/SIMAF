"""
Fact Checking Module for Knowledge Validation Framework.

Provides fact-checking, authority scoring, and currency tracking for knowledge claims.
Validates information against multiple sources and maintains confidence scores.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import hashlib
import json


logger = logging.getLogger(__name__)


class FactStatus(Enum):
    """Status of a fact after validation."""
    VERIFIED = "verified"
    DISPUTED = "disputed"
    UNVERIFIED = "unverified"
    OUTDATED = "outdated"
    CONFLICTING = "conflicting"


class SourceType(Enum):
    """Types of knowledge sources."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    EXPERT = "expert"
    CROWD = "crowd"
    AI_GENERATED = "ai_generated"
    DOCUMENTATION = "documentation"
    CODE_ANALYSIS = "code_analysis"


@dataclass
class AuthorityScore:
    """Authority score for a knowledge source."""
    source_id: str
    source_type: SourceType
    base_score: float = 0.5
    expertise_score: float = 0.5
    reliability_score: float = 0.5
    recency_score: float = 0.5
    citation_count: int = 0
    verification_count: int = 0
    dispute_count: int = 0

    @property
    def composite_score(self) -> float:
        """Calculate composite authority score."""
        weights = {
            'base': 0.2,
            'expertise': 0.3,
            'reliability': 0.3,
            'recency': 0.2
        }

        score = (
            weights['base'] * self.base_score +
            weights['expertise'] * self.expertise_score +
            weights['reliability'] * self.reliability_score +
            weights['recency'] * self.recency_score
        )

        citation_boost = min(0.1, self.citation_count * 0.01)
        verification_boost = min(0.1, self.verification_count * 0.02)
        dispute_penalty = min(0.2, self.dispute_count * 0.05)

        return max(0.0, min(1.0, score + citation_boost + verification_boost - dispute_penalty))


@dataclass
class FactClaim:
    """Represents a factual claim to be validated."""
    claim_id: str
    content: str
    source_id: str
    timestamp: datetime
    domain: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate claim_id if not provided."""
        if not self.claim_id:
            claim_hash = hashlib.sha256(
                f"{self.content}:{self.source_id}".encode()
            ).hexdigest()[:16]
            self.claim_id = f"claim_{claim_hash}"


@dataclass
class ValidationResult:
    """Result of fact validation."""
    claim_id: str
    status: FactStatus
    confidence: float
    supporting_sources: List[str] = field(default_factory=list)
    disputing_sources: List[str] = field(default_factory=list)
    verification_timestamp: datetime = field(default_factory=datetime.now)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    explanation: str = ""

    @property
    def agreement_ratio(self) -> float:
        """Calculate ratio of supporting to total sources."""
        total = len(self.supporting_sources) + len(self.disputing_sources)
        if total == 0:
            return 0.0
        return len(self.supporting_sources) / total


class FactChecker:
    """
    Main fact-checking engine for validating knowledge claims.

    Validates facts against multiple sources, tracks authority scores,
    and maintains currency of information.
    """

    def __init__(
        self,
        currency_threshold_days: int = 90,
        min_confidence_threshold: float = 0.6,
        min_sources_required: int = 2
    ):
        """
        Initialize FactChecker.

        Args:
            currency_threshold_days: Days before information is considered outdated
            min_confidence_threshold: Minimum confidence for verification
            min_sources_required: Minimum sources needed for validation
        """
        self.currency_threshold_days = currency_threshold_days
        self.min_confidence_threshold = min_confidence_threshold
        self.min_sources_required = min_sources_required

        self.authority_scores: Dict[str, AuthorityScore] = {}
        self.fact_claims: Dict[str, FactClaim] = {}
        self.validation_results: Dict[str, ValidationResult] = {}
        self.source_reliability_history: Dict[str, List[float]] = defaultdict(list)

        logger.info(
            f"FactChecker initialized with currency_threshold={currency_threshold_days}d, "
            f"min_confidence={min_confidence_threshold}, min_sources={min_sources_required}"
        )

    def register_source(
        self,
        source_id: str,
        source_type: SourceType,
        base_score: float = 0.5,
        expertise_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuthorityScore:
        """
        Register a knowledge source with initial authority score.

        Args:
            source_id: Unique identifier for source
            source_type: Type of source
            base_score: Initial base authority score (0-1)
            expertise_score: Initial expertise score (0-1)
            metadata: Additional source metadata

        Returns:
            AuthorityScore object for the source
        """
        if source_id in self.authority_scores:
            logger.warning(f"Source {source_id} already registered, updating scores")

        authority = AuthorityScore(
            source_id=source_id,
            source_type=source_type,
            base_score=base_score,
            expertise_score=expertise_score
        )

        self.authority_scores[source_id] = authority
        logger.info(f"Registered source {source_id} with type {source_type.value}")

        return authority

    def update_source_reliability(
        self,
        source_id: str,
        reliability_score: float,
        recency_score: Optional[float] = None
    ) -> None:
        """
        Update reliability score for a source.

        Args:
            source_id: Source to update
            reliability_score: New reliability score (0-1)
            recency_score: Optional recency score (0-1)
        """
        if source_id not in self.authority_scores:
            logger.warning(f"Cannot update unknown source {source_id}")
            return

        authority = self.authority_scores[source_id]
        authority.reliability_score = max(0.0, min(1.0, reliability_score))

        if recency_score is not None:
            authority.recency_score = max(0.0, min(1.0, recency_score))

        self.source_reliability_history[source_id].append(reliability_score)

        logger.debug(
            f"Updated source {source_id} reliability to {reliability_score:.3f}"
        )

    def submit_claim(
        self,
        content: str,
        source_id: str,
        timestamp: Optional[datetime] = None,
        domain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FactClaim:
        """
        Submit a fact claim for validation.

        Args:
            content: The factual claim content
            source_id: Source making the claim
            timestamp: When claim was made
            domain: Knowledge domain
            metadata: Additional metadata

        Returns:
            FactClaim object
        """
        if timestamp is None:
            timestamp = datetime.now()

        claim = FactClaim(
            claim_id="",
            content=content,
            source_id=source_id,
            timestamp=timestamp,
            domain=domain,
            metadata=metadata or {}
        )

        self.fact_claims[claim.claim_id] = claim
        logger.info(f"Submitted claim {claim.claim_id} from source {source_id}")

        return claim

    def check_currency(self, claim: FactClaim) -> Tuple[bool, float]:
        """
        Check if a claim is current based on age.

        Args:
            claim: Claim to check

        Returns:
            Tuple of (is_current, currency_score)
        """
        age = datetime.now() - claim.timestamp
        age_days = age.total_seconds() / 86400

        if age_days > self.currency_threshold_days:
            currency_score = max(0.0, 1.0 - (age_days / (self.currency_threshold_days * 2)))
            return False, currency_score

        currency_score = 1.0 - (age_days / self.currency_threshold_days)
        return True, max(0.0, min(1.0, currency_score))

    def validate_claim(
        self,
        claim_id: str,
        supporting_sources: Optional[List[str]] = None,
        disputing_sources: Optional[List[str]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None
    ) -> ValidationResult:
        """
        Validate a fact claim against sources.

        Args:
            claim_id: ID of claim to validate
            supporting_sources: Sources supporting the claim
            disputing_sources: Sources disputing the claim
            evidence: Supporting evidence

        Returns:
            ValidationResult with validation outcome
        """
        if claim_id not in self.fact_claims:
            logger.error(f"Unknown claim {claim_id}")
            raise ValueError(f"Claim {claim_id} not found")

        claim = self.fact_claims[claim_id]
        supporting_sources = supporting_sources or []
        disputing_sources = disputing_sources or []
        evidence = evidence or []

        is_current, currency_score = self.check_currency(claim)

        supporting_authority = self._calculate_aggregate_authority(supporting_sources)
        disputing_authority = self._calculate_aggregate_authority(disputing_sources)

        total_sources = len(supporting_sources) + len(disputing_sources)

        if total_sources < self.min_sources_required:
            status = FactStatus.UNVERIFIED
            confidence = 0.3
            explanation = f"Insufficient sources ({total_sources} < {self.min_sources_required})"
        elif not is_current:
            status = FactStatus.OUTDATED
            confidence = currency_score * 0.5
            explanation = f"Information outdated (age > {self.currency_threshold_days} days)"
        elif len(supporting_sources) > 0 and len(disputing_sources) > 0:
            status = FactStatus.CONFLICTING
            confidence = abs(supporting_authority - disputing_authority)
            explanation = "Sources provide conflicting information"
        elif supporting_authority > disputing_authority:
            confidence = supporting_authority * currency_score
            if confidence >= self.min_confidence_threshold:
                status = FactStatus.VERIFIED
                explanation = f"Verified with confidence {confidence:.2f}"
            else:
                status = FactStatus.UNVERIFIED
                explanation = f"Confidence {confidence:.2f} below threshold"
        else:
            status = FactStatus.DISPUTED
            confidence = disputing_authority * currency_score
            explanation = "Claim disputed by authoritative sources"

        result = ValidationResult(
            claim_id=claim_id,
            status=status,
            confidence=confidence,
            supporting_sources=supporting_sources,
            disputing_sources=disputing_sources,
            evidence=evidence,
            explanation=explanation
        )

        self.validation_results[claim_id] = result
        self._update_source_citations(supporting_sources, disputing_sources)

        logger.info(
            f"Validated claim {claim_id}: {status.value} "
            f"(confidence={confidence:.3f})"
        )

        return result

    def _calculate_aggregate_authority(self, source_ids: List[str]) -> float:
        """
        Calculate aggregate authority score from multiple sources.

        Args:
            source_ids: List of source IDs

        Returns:
            Aggregate authority score (0-1)
        """
        if not source_ids:
            return 0.0

        valid_scores = []
        for source_id in source_ids:
            if source_id in self.authority_scores:
                score = self.authority_scores[source_id].composite_score
                valid_scores.append(score)
            else:
                logger.warning(f"Unknown source {source_id} in validation")

        if not valid_scores:
            return 0.0

        weighted_avg = sum(valid_scores) / len(valid_scores)

        diversity_bonus = min(0.1, len(valid_scores) * 0.02)

        return min(1.0, weighted_avg + diversity_bonus)

    def _update_source_citations(
        self,
        supporting_sources: List[str],
        disputing_sources: List[str]
    ) -> None:
        """
        Update citation and verification counts for sources.

        Args:
            supporting_sources: Sources that supported claim
            disputing_sources: Sources that disputed claim
        """
        for source_id in supporting_sources:
            if source_id in self.authority_scores:
                self.authority_scores[source_id].citation_count += 1
                self.authority_scores[source_id].verification_count += 1

        for source_id in disputing_sources:
            if source_id in self.authority_scores:
                self.authority_scores[source_id].citation_count += 1
                self.authority_scores[source_id].dispute_count += 1

    def get_source_authority(self, source_id: str) -> Optional[AuthorityScore]:
        """
        Get authority score for a source.

        Args:
            source_id: Source to query

        Returns:
            AuthorityScore or None if not found
        """
        return self.authority_scores.get(source_id)

    def get_validation_result(self, claim_id: str) -> Optional[ValidationResult]:
        """
        Get validation result for a claim.

        Args:
            claim_id: Claim to query

        Returns:
            ValidationResult or None if not found
        """
        return self.validation_results.get(claim_id)

    def get_verified_claims(
        self,
        min_confidence: Optional[float] = None,
        domain: Optional[str] = None
    ) -> List[Tuple[FactClaim, ValidationResult]]:
        """
        Get all verified claims matching criteria.

        Args:
            min_confidence: Minimum confidence threshold
            domain: Filter by domain

        Returns:
            List of (claim, result) tuples
        """
        min_conf = min_confidence or self.min_confidence_threshold
        verified = []

        for claim_id, result in self.validation_results.items():
            if result.status != FactStatus.VERIFIED:
                continue
            if result.confidence < min_conf:
                continue

            claim = self.fact_claims.get(claim_id)
            if not claim:
                continue

            if domain and claim.domain != domain:
                continue

            verified.append((claim, result))

        return sorted(verified, key=lambda x: x[1].confidence, reverse=True)

    def export_state(self) -> Dict[str, Any]:
        """
        Export fact checker state for persistence.

        Returns:
            Dictionary with complete state
        """
        return {
            'authority_scores': {
                sid: {
                    'source_id': auth.source_id,
                    'source_type': auth.source_type.value,
                    'base_score': auth.base_score,
                    'expertise_score': auth.expertise_score,
                    'reliability_score': auth.reliability_score,
                    'recency_score': auth.recency_score,
                    'citation_count': auth.citation_count,
                    'verification_count': auth.verification_count,
                    'dispute_count': auth.dispute_count
                }
                for sid, auth in self.authority_scores.items()
            },
            'fact_claims': {
                cid: {
                    'claim_id': claim.claim_id,
                    'content': claim.content,
                    'source_id': claim.source_id,
                    'timestamp': claim.timestamp.isoformat(),
                    'domain': claim.domain,
                    'metadata': claim.metadata
                }
                for cid, claim in self.fact_claims.items()
            },
            'validation_results': {
                cid: {
                    'claim_id': result.claim_id,
                    'status': result.status.value,
                    'confidence': result.confidence,
                    'supporting_sources': result.supporting_sources,
                    'disputing_sources': result.disputing_sources,
                    'verification_timestamp': result.verification_timestamp.isoformat(),
                    'evidence': result.evidence,
                    'explanation': result.explanation
                }
                for cid, result in self.validation_results.items()
            },
            'config': {
                'currency_threshold_days': self.currency_threshold_days,
                'min_confidence_threshold': self.min_confidence_threshold,
                'min_sources_required': self.min_sources_required
            }
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """
        Import fact checker state from persistence.

        Args:
            state: State dictionary from export_state
        """
        if 'config' in state:
            config = state['config']
            self.currency_threshold_days = config.get('currency_threshold_days', 90)
            self.min_confidence_threshold = config.get('min_confidence_threshold', 0.6)
            self.min_sources_required = config.get('min_sources_required', 2)

        if 'authority_scores' in state:
            for sid, auth_data in state['authority_scores'].items():
                authority = AuthorityScore(
                    source_id=auth_data['source_id'],
                    source_type=SourceType(auth_data['source_type']),
                    base_score=auth_data['base_score'],
                    expertise_score=auth_data['expertise_score'],
                    reliability_score=auth_data['reliability_score'],
                    recency_score=auth_data['recency_score'],
                    citation_count=auth_data['citation_count'],
                    verification_count=auth_data['verification_count'],
                    dispute_count=auth_data['dispute_count']
                )
                self.authority_scores[sid] = authority

        if 'fact_claims' in state:
            for cid, claim_data in state['fact_claims'].items():
                claim = FactClaim(
                    claim_id=claim_data['claim_id'],
                    content=claim_data['content'],
                    source_id=claim_data['source_id'],
                    timestamp=datetime.fromisoformat(claim_data['timestamp']),
                    domain=claim_data.get('domain'),
                    metadata=claim_data.get('metadata', {})
                )
                self.fact_claims[cid] = claim

        if 'validation_results' in state:
            for cid, result_data in state['validation_results'].items():
                result = ValidationResult(
                    claim_id=result_data['claim_id'],
                    status=FactStatus(result_data['status']),
                    confidence=result_data['confidence'],
                    supporting_sources=result_data['supporting_sources'],
                    disputing_sources=result_data['disputing_sources'],
                    verification_timestamp=datetime.fromisoformat(
                        result_data['verification_timestamp']
                    ),
                    evidence=result_data.get('evidence', []),
                    explanation=result_data.get('explanation', '')
                )
                self.validation_results[cid] = result

        logger.info(f"Imported state with {len(self.authority_scores)} sources")
