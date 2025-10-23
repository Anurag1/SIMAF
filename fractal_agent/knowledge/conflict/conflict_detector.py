"""
Conflict Detection Module for Knowledge Validation Framework.

Detects and resolves conflicts between knowledge sources, tracks inconsistencies,
and provides resolution strategies based on authority and evidence.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import hashlib


logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of knowledge conflicts."""
    DIRECT_CONTRADICTION = "direct_contradiction"
    PARTIAL_DISAGREEMENT = "partial_disagreement"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    SCOPE_MISMATCH = "scope_mismatch"
    DEFINITION_CONFLICT = "definition_conflict"
    VALUE_DISCREPANCY = "value_discrepancy"
    SEMANTIC_CONFLICT = "semantic_conflict"


class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts."""
    HIGHEST_AUTHORITY = "highest_authority"
    MOST_RECENT = "most_recent"
    MAJORITY_CONSENSUS = "majority_consensus"
    EVIDENCE_BASED = "evidence_based"
    CONTEXT_DEPENDENT = "context_dependent"
    MANUAL_REVIEW = "manual_review"


class ConflictSeverity(Enum):
    """Severity levels for conflicts."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


@dataclass
class KnowledgeStatement:
    """Represents a knowledge statement from a source."""
    statement_id: str
    content: str
    source_id: str
    timestamp: datetime
    domain: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    authority_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate statement_id if not provided."""
        if not self.statement_id:
            stmt_hash = hashlib.sha256(
                f"{self.content}:{self.source_id}:{self.timestamp}".encode()
            ).hexdigest()[:16]
            self.statement_id = f"stmt_{stmt_hash}"


@dataclass
class Conflict:
    """Represents a detected conflict between knowledge statements."""
    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    statements: List[str]
    description: str
    detected_at: datetime = field(default_factory=datetime.now)
    resolution_strategy: Optional[ResolutionStrategy] = None
    resolved: bool = False
    resolution: Optional[str] = None
    resolution_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate conflict_id if not provided."""
        if not self.conflict_id:
            conflict_hash = hashlib.sha256(
                f"{':'.join(sorted(self.statements))}:{self.conflict_type.value}".encode()
            ).hexdigest()[:16]
            self.conflict_id = f"conflict_{conflict_hash}"


@dataclass
class ResolutionResult:
    """Result of conflict resolution."""
    conflict_id: str
    strategy_used: ResolutionStrategy
    selected_statement: Optional[str]
    confidence: float
    reasoning: str
    supporting_evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class ConflictDetector:
    """
    Main conflict detection engine for knowledge validation.

    Detects conflicts between knowledge statements, assesses severity,
    and provides resolution recommendations based on configured strategies.
    """

    def __init__(
        self,
        default_strategy: ResolutionStrategy = ResolutionStrategy.HIGHEST_AUTHORITY,
        auto_resolve_threshold: float = 0.8,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize ConflictDetector.

        Args:
            default_strategy: Default resolution strategy
            auto_resolve_threshold: Confidence threshold for auto-resolution
            similarity_threshold: Threshold for detecting similar statements
        """
        self.default_strategy = default_strategy
        self.auto_resolve_threshold = auto_resolve_threshold
        self.similarity_threshold = similarity_threshold

        self.statements: Dict[str, KnowledgeStatement] = {}
        self.conflicts: Dict[str, Conflict] = {}
        self.statement_index: Dict[str, Set[str]] = defaultdict(set)
        self.domain_index: Dict[str, Set[str]] = defaultdict(set)
        self.source_index: Dict[str, Set[str]] = defaultdict(set)

        logger.info(
            f"ConflictDetector initialized with strategy={default_strategy.value}, "
            f"auto_resolve_threshold={auto_resolve_threshold}"
        )

    def add_statement(
        self,
        content: str,
        source_id: str,
        timestamp: Optional[datetime] = None,
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        authority_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgeStatement:
        """
        Add a knowledge statement to the conflict detection system.

        Args:
            content: Statement content
            source_id: Source making the statement
            timestamp: When statement was made
            domain: Knowledge domain
            context: Contextual information
            authority_score: Authority score of source
            metadata: Additional metadata

        Returns:
            KnowledgeStatement object
        """
        if timestamp is None:
            timestamp = datetime.now()

        statement = KnowledgeStatement(
            statement_id="",
            content=content,
            source_id=source_id,
            timestamp=timestamp,
            domain=domain,
            context=context or {},
            authority_score=authority_score,
            metadata=metadata or {}
        )

        self.statements[statement.statement_id] = statement

        content_key = self._generate_content_key(content)
        self.statement_index[content_key].add(statement.statement_id)

        if domain:
            self.domain_index[domain].add(statement.statement_id)

        self.source_index[source_id].add(statement.statement_id)

        self._check_for_conflicts(statement)

        logger.info(
            f"Added statement {statement.statement_id} from source {source_id}"
        )

        return statement

    def _generate_content_key(self, content: str) -> str:
        """
        Generate normalized key for content indexing.

        Args:
            content: Statement content

        Returns:
            Normalized content key
        """
        normalized = content.lower().strip()
        key_words = set(normalized.split())
        sorted_words = sorted(key_words)
        return '_'.join(sorted_words[:10])

    def _check_for_conflicts(self, new_statement: KnowledgeStatement) -> None:
        """
        Check if new statement conflicts with existing statements.

        Args:
            new_statement: Statement to check
        """
        content_key = self._generate_content_key(new_statement.content)

        candidates = set()
        if content_key in self.statement_index:
            candidates.update(self.statement_index[content_key])

        if new_statement.domain:
            candidates.update(self.domain_index.get(new_statement.domain, set()))

        for stmt_id in candidates:
            if stmt_id == new_statement.statement_id:
                continue

            existing = self.statements[stmt_id]
            conflict_type = self._detect_conflict_type(new_statement, existing)

            if conflict_type:
                self._create_conflict(new_statement, existing, conflict_type)

    def _detect_conflict_type(
        self,
        stmt1: KnowledgeStatement,
        stmt2: KnowledgeStatement
    ) -> Optional[ConflictType]:
        """
        Detect type of conflict between two statements.

        Args:
            stmt1: First statement
            stmt2: Second statement

        Returns:
            ConflictType if conflict detected, None otherwise
        """
        if stmt1.source_id == stmt2.source_id:
            return None

        if stmt1.domain != stmt2.domain:
            return None

        content1_lower = stmt1.content.lower()
        content2_lower = stmt2.content.lower()

        negation_words = {'not', 'no', 'never', 'none', 'neither', 'nobody', 'nothing'}
        has_negation_1 = any(word in content1_lower.split() for word in negation_words)
        has_negation_2 = any(word in content2_lower.split() for word in negation_words)

        word_overlap = self._calculate_word_overlap(content1_lower, content2_lower)

        if word_overlap > 0.5 and has_negation_1 != has_negation_2:
            return ConflictType.DIRECT_CONTRADICTION

        time_diff = abs((stmt1.timestamp - stmt2.timestamp).total_seconds())
        if time_diff < 3600 and word_overlap > 0.6:
            return ConflictType.TEMPORAL_INCONSISTENCY

        if self._has_numeric_discrepancy(content1_lower, content2_lower):
            return ConflictType.VALUE_DISCREPANCY

        if word_overlap > 0.4:
            return ConflictType.PARTIAL_DISAGREEMENT

        return None

    def _calculate_word_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate word overlap ratio between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Overlap ratio (0-1)
        """
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _has_numeric_discrepancy(self, text1: str, text2: str) -> bool:
        """
        Check if texts contain numeric discrepancies.

        Args:
            text1: First text
            text2: Second text

        Returns:
            True if numeric discrepancy detected
        """
        import re

        numbers1 = set(re.findall(r'\b\d+(?:\.\d+)?\b', text1))
        numbers2 = set(re.findall(r'\b\d+(?:\.\d+)?\b', text2))

        if not numbers1 or not numbers2:
            return False

        return len(numbers1.intersection(numbers2)) == 0

    def _create_conflict(
        self,
        stmt1: KnowledgeStatement,
        stmt2: KnowledgeStatement,
        conflict_type: ConflictType
    ) -> Conflict:
        """
        Create conflict record for two statements.

        Args:
            stmt1: First statement
            stmt2: Second statement
            conflict_type: Type of conflict

        Returns:
            Conflict object
        """
        severity = self._assess_severity(stmt1, stmt2, conflict_type)

        description = (
            f"{conflict_type.value}: "
            f"Statement from {stmt1.source_id} conflicts with {stmt2.source_id}"
        )

        conflict = Conflict(
            conflict_id="",
            conflict_type=conflict_type,
            severity=severity,
            statements=[stmt1.statement_id, stmt2.statement_id],
            description=description
        )

        existing_conflict = self._find_existing_conflict(
            stmt1.statement_id, stmt2.statement_id
        )

        if existing_conflict:
            logger.debug(
                f"Conflict already exists: {existing_conflict.conflict_id}"
            )
            return existing_conflict

        self.conflicts[conflict.conflict_id] = conflict

        logger.info(
            f"Created conflict {conflict.conflict_id}: {conflict_type.value} "
            f"(severity={severity.value})"
        )

        if self._should_auto_resolve(conflict):
            self.resolve_conflict(conflict.conflict_id)

        return conflict

    def _find_existing_conflict(self, stmt_id1: str, stmt_id2: str) -> Optional[Conflict]:
        """
        Find existing conflict between two statements.

        Args:
            stmt_id1: First statement ID
            stmt_id2: Second statement ID

        Returns:
            Existing Conflict or None
        """
        for conflict in self.conflicts.values():
            if set([stmt_id1, stmt_id2]).issubset(set(conflict.statements)):
                return conflict
        return None

    def _assess_severity(
        self,
        stmt1: KnowledgeStatement,
        stmt2: KnowledgeStatement,
        conflict_type: ConflictType
    ) -> ConflictSeverity:
        """
        Assess severity of conflict.

        Args:
            stmt1: First statement
            stmt2: Second statement
            conflict_type: Type of conflict

        Returns:
            ConflictSeverity level
        """
        authority_diff = abs(stmt1.authority_score - stmt2.authority_score)

        if conflict_type == ConflictType.DIRECT_CONTRADICTION:
            if authority_diff < 0.1:
                return ConflictSeverity.CRITICAL
            return ConflictSeverity.HIGH

        if conflict_type == ConflictType.VALUE_DISCREPANCY:
            return ConflictSeverity.HIGH

        if conflict_type == ConflictType.TEMPORAL_INCONSISTENCY:
            return ConflictSeverity.MEDIUM

        if authority_diff > 0.5:
            return ConflictSeverity.LOW

        return ConflictSeverity.MEDIUM

    def _should_auto_resolve(self, conflict: Conflict) -> bool:
        """
        Determine if conflict should be auto-resolved.

        Args:
            conflict: Conflict to evaluate

        Returns:
            True if should auto-resolve
        """
        if conflict.severity in [ConflictSeverity.CRITICAL, ConflictSeverity.HIGH]:
            return False

        if len(conflict.statements) > 2:
            return True

        if conflict.conflict_type == ConflictType.TEMPORAL_INCONSISTENCY:
            return True

        return False

    def resolve_conflict(
        self,
        conflict_id: str,
        strategy: Optional[ResolutionStrategy] = None
    ) -> ResolutionResult:
        """
        Resolve a conflict using specified or default strategy.

        Args:
            conflict_id: Conflict to resolve
            strategy: Resolution strategy to use

        Returns:
            ResolutionResult with resolution details
        """
        if conflict_id not in self.conflicts:
            raise ValueError(f"Conflict {conflict_id} not found")

        conflict = self.conflicts[conflict_id]
        strategy = strategy or self.default_strategy

        statements = [
            self.statements[stmt_id]
            for stmt_id in conflict.statements
            if stmt_id in self.statements
        ]

        if not statements:
            raise ValueError(f"No valid statements for conflict {conflict_id}")

        if strategy == ResolutionStrategy.HIGHEST_AUTHORITY:
            result = self._resolve_by_authority(conflict, statements)
        elif strategy == ResolutionStrategy.MOST_RECENT:
            result = self._resolve_by_recency(conflict, statements)
        elif strategy == ResolutionStrategy.MAJORITY_CONSENSUS:
            result = self._resolve_by_consensus(conflict, statements)
        elif strategy == ResolutionStrategy.EVIDENCE_BASED:
            result = self._resolve_by_evidence(conflict, statements)
        else:
            result = ResolutionResult(
                conflict_id=conflict_id,
                strategy_used=ResolutionStrategy.MANUAL_REVIEW,
                selected_statement=None,
                confidence=0.0,
                reasoning="Requires manual review"
            )

        if result.confidence >= self.auto_resolve_threshold:
            conflict.resolved = True
            conflict.resolution = result.selected_statement
            conflict.resolution_strategy = strategy
            conflict.resolution_timestamp = datetime.now()

            logger.info(
                f"Resolved conflict {conflict_id} using {strategy.value} "
                f"(confidence={result.confidence:.3f})"
            )
        else:
            logger.info(
                f"Resolution confidence {result.confidence:.3f} below threshold, "
                f"conflict {conflict_id} requires manual review"
            )

        return result

    def _resolve_by_authority(
        self,
        conflict: Conflict,
        statements: List[KnowledgeStatement]
    ) -> ResolutionResult:
        """Resolve conflict by highest authority score."""
        best_stmt = max(statements, key=lambda s: s.authority_score)

        authority_gap = best_stmt.authority_score - min(
            s.authority_score for s in statements if s != best_stmt
        )

        confidence = min(1.0, best_stmt.authority_score * (1 + authority_gap))

        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            strategy_used=ResolutionStrategy.HIGHEST_AUTHORITY,
            selected_statement=best_stmt.statement_id,
            confidence=confidence,
            reasoning=f"Selected statement from highest authority source "
                     f"(score={best_stmt.authority_score:.3f})"
        )

    def _resolve_by_recency(
        self,
        conflict: Conflict,
        statements: List[KnowledgeStatement]
    ) -> ResolutionResult:
        """Resolve conflict by most recent statement."""
        most_recent = max(statements, key=lambda s: s.timestamp)

        time_gap = (most_recent.timestamp - min(
            s.timestamp for s in statements if s != most_recent
        )).total_seconds()

        recency_score = min(1.0, time_gap / 86400 / 30)
        confidence = (most_recent.authority_score + recency_score) / 2

        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            strategy_used=ResolutionStrategy.MOST_RECENT,
            selected_statement=most_recent.statement_id,
            confidence=confidence,
            reasoning=f"Selected most recent statement from {most_recent.timestamp}"
        )

    def _resolve_by_consensus(
        self,
        conflict: Conflict,
        statements: List[KnowledgeStatement]
    ) -> ResolutionResult:
        """Resolve conflict by majority consensus."""
        content_groups = defaultdict(list)
        for stmt in statements:
            key = self._generate_content_key(stmt.content)
            content_groups[key].append(stmt)

        largest_group = max(content_groups.values(), key=len)
        best_in_group = max(largest_group, key=lambda s: s.authority_score)

        consensus_ratio = len(largest_group) / len(statements)
        confidence = consensus_ratio * best_in_group.authority_score

        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            strategy_used=ResolutionStrategy.MAJORITY_CONSENSUS,
            selected_statement=best_in_group.statement_id,
            confidence=confidence,
            reasoning=f"Selected by consensus ({len(largest_group)}/{len(statements)} sources)"
        )

    def _resolve_by_evidence(
        self,
        conflict: Conflict,
        statements: List[KnowledgeStatement]
    ) -> ResolutionResult:
        """Resolve conflict by evidence strength."""
        evidence_scores = []
        for stmt in statements:
            evidence_count = len(stmt.metadata.get('evidence', []))
            evidence_quality = stmt.metadata.get('evidence_quality', 0.5)
            evidence_score = (evidence_count * 0.1 + evidence_quality) * stmt.authority_score
            evidence_scores.append((stmt, evidence_score))

        best_stmt, best_score = max(evidence_scores, key=lambda x: x[1])

        confidence = min(1.0, best_score)

        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            strategy_used=ResolutionStrategy.EVIDENCE_BASED,
            selected_statement=best_stmt.statement_id,
            confidence=confidence,
            reasoning=f"Selected based on evidence strength (score={best_score:.3f})"
        )

    def get_conflicts(
        self,
        domain: Optional[str] = None,
        severity: Optional[ConflictSeverity] = None,
        resolved: Optional[bool] = None
    ) -> List[Conflict]:
        """
        Get conflicts matching criteria.

        Args:
            domain: Filter by domain
            severity: Filter by severity
            resolved: Filter by resolution status

        Returns:
            List of matching conflicts
        """
        conflicts = []

        for conflict in self.conflicts.values():
            if resolved is not None and conflict.resolved != resolved:
                continue

            if severity and conflict.severity != severity:
                continue

            if domain:
                stmt_domains = [
                    self.statements[sid].domain
                    for sid in conflict.statements
                    if sid in self.statements
                ]
                if domain not in stmt_domains:
                    continue

            conflicts.append(conflict)

        return sorted(
            conflicts,
            key=lambda c: (c.severity.value, c.detected_at),
            reverse=True
        )

    def get_statement(self, statement_id: str) -> Optional[KnowledgeStatement]:
        """
        Get statement by ID.

        Args:
            statement_id: Statement to retrieve

        Returns:
            KnowledgeStatement or None if not found
        """
        return self.statements.get(statement_id)

    def export_state(self) -> Dict[str, Any]:
        """
        Export conflict detector state for persistence.

        Returns:
            Dictionary with complete state
        """
        return {
            'statements': {
                sid: {
                    'statement_id': stmt.statement_id,
                    'content': stmt.content,
                    'source_id': stmt.source_id,
                    'timestamp': stmt.timestamp.isoformat(),
                    'domain': stmt.domain,
                    'context': stmt.context,
                    'authority_score': stmt.authority_score,
                    'metadata': stmt.metadata
                }
                for sid, stmt in self.statements.items()
            },
            'conflicts': {
                cid: {
                    'conflict_id': conflict.conflict_id,
                    'conflict_type': conflict.conflict_type.value,
                    'severity': conflict.severity.value,
                    'statements': conflict.statements,
                    'description': conflict.description,
                    'detected_at': conflict.detected_at.isoformat(),
                    'resolution_strategy': conflict.resolution_strategy.value if conflict.resolution_strategy else None,
                    'resolved': conflict.resolved,
                    'resolution': conflict.resolution,
                    'resolution_timestamp': conflict.resolution_timestamp.isoformat() if conflict.resolution_timestamp else None,
                    'metadata': conflict.metadata
                }
                for cid, conflict in self.conflicts.items()
            },
            'config': {
                'default_strategy': self.default_strategy.value,
                'auto_resolve_threshold': self.auto_resolve_threshold,
                'similarity_threshold': self.similarity_threshold
            }
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """
        Import conflict detector state from persistence.

        Args:
            state: State dictionary from export_state
        """
        if 'config' in state:
            config = state['config']
            self.default_strategy = ResolutionStrategy(config['default_strategy'])
            self.auto_resolve_threshold = config.get('auto_resolve_threshold', 0.8)
            self.similarity_threshold = config.get('similarity_threshold', 0.7)

        if 'statements' in state:
            for sid, stmt_data in state['statements'].items():
                stmt = KnowledgeStatement(
                    statement_id=stmt_data['statement_id'],
                    content=stmt_data['content'],
                    source_id=stmt_data['source_id'],
                    timestamp=datetime.fromisoformat(stmt_data['timestamp']),
                    domain=stmt_data.get('domain'),
                    context=stmt_data.get('context', {}),
                    authority_score=stmt_data.get('authority_score', 0.5),
                    metadata=stmt_data.get('metadata', {})
                )
                self.statements[sid] = stmt

        if 'conflicts' in state:
            for cid, conflict_data in state['conflicts'].items():
                conflict = Conflict(
                    conflict_id=conflict_data['conflict_id'],
                    conflict_type=ConflictType(conflict_data['conflict_type']),
                    severity=ConflictSeverity(conflict_data['severity']),
                    statements=conflict_data['statements'],
                    description=conflict_data['description'],
                    detected_at=datetime.fromisoformat(conflict_data['detected_at']),
                    resolution_strategy=ResolutionStrategy(conflict_data['resolution_strategy']) if conflict_data.get('resolution_strategy') else None,
                    resolved=conflict_data.get('resolved', False),
                    resolution=conflict_data.get('resolution'),
                    resolution_timestamp=datetime.fromisoformat(conflict_data['resolution_timestamp']) if conflict_data.get('resolution_timestamp') else None,
                    metadata=conflict_data.get('metadata', {})
                )
                self.conflicts[cid] = conflict

        logger.info(
            f"Imported state with {len(self.statements)} statements "
            f"and {len(self.conflicts)} conflicts"
        )
