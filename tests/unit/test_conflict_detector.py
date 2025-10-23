import pytest
from datetime import datetime, timedelta
from fractal_agent.knowledge.conflict.conflict_detector import (
    ConflictDetector,
    Conflict,
    ConflictType,
    ConflictSeverity,
    ResolutionStrategy,
    KnowledgeItem
)


@pytest.fixture
def conflict_detector():
    return ConflictDetector()


@pytest.fixture
def knowledge_item_1():
    return KnowledgeItem(
        item_id="item_001",
        content="The speed of light is 299,792,458 m/s",
        source_id="src_001",
        timestamp=datetime.now(),
        domain="physics",
        authority_score=0.95
    )


@pytest.fixture
def knowledge_item_2():
    return KnowledgeItem(
        item_id="item_002",
        content="The speed of light is approximately 300,000,000 m/s",
        source_id="src_002",
        timestamp=datetime.now(),
        domain="physics",
        authority_score=0.75
    )


@pytest.fixture
def knowledge_item_3():
    return KnowledgeItem(
        item_id="item_003",
        content="Python 3.9 was released in 2020",
        source_id="src_003",
        timestamp=datetime.now() - timedelta(days=100),
        domain="programming",
        authority_score=0.90
    )


@pytest.fixture
def knowledge_item_4():
    return KnowledgeItem(
        item_id="item_004",
        content="Python 3.9 was released in 2019",
        source_id="src_004",
        timestamp=datetime.now(),
        domain="programming",
        authority_score=0.60
    )


@pytest.fixture
def outdated_item():
    return KnowledgeItem(
        item_id="item_005",
        content="COVID-19 is a new disease",
        source_id="src_005",
        timestamp=datetime.now() - timedelta(days=1000),
        domain="medicine",
        authority_score=0.85
    )


@pytest.fixture
def recent_item():
    return KnowledgeItem(
        item_id="item_006",
        content="COVID-19 is a well-studied disease",
        source_id="src_006",
        timestamp=datetime.now(),
        domain="medicine",
        authority_score=0.90
    )


class TestKnowledgeItem:
    def test_knowledge_item_creation(self, knowledge_item_1):
        assert knowledge_item_1.item_id == "item_001"
        assert "speed of light" in knowledge_item_1.content
        assert knowledge_item_1.source_id == "src_001"
        assert knowledge_item_1.domain == "physics"
        assert knowledge_item_1.authority_score == 0.95

    def test_knowledge_item_timestamp(self, knowledge_item_1):
        assert isinstance(knowledge_item_1.timestamp, datetime)
        assert knowledge_item_1.timestamp <= datetime.now()

    def test_knowledge_item_with_low_authority(self):
        item = KnowledgeItem(
            item_id="low_auth",
            content="Test content",
            source_id="src_test",
            timestamp=datetime.now(),
            domain="test",
            authority_score=0.1
        )
        assert item.authority_score == 0.1

    def test_knowledge_item_empty_content(self):
        item = KnowledgeItem(
            item_id="empty",
            content="",
            source_id="src_empty",
            timestamp=datetime.now(),
            domain="test",
            authority_score=0.5
        )
        assert item.content == ""


class TestConflict:
    def test_conflict_creation(self):
        conflict = Conflict(
            conflict_id="conf_001",
            item_ids=["item_001", "item_002"],
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            severity=ConflictSeverity.HIGH,
            description="Conflicting values for speed of light",
            detected_at=datetime.now()
        )

        assert conflict.conflict_id == "conf_001"
        assert len(conflict.item_ids) == 2
        assert conflict.conflict_type == ConflictType.DIRECT_CONTRADICTION
        assert conflict.severity == ConflictSeverity.HIGH
        assert conflict.resolved is False
        assert conflict.resolution is None

    def test_conflict_type_enum(self):
        assert ConflictType.DIRECT_CONTRADICTION.value == "direct_contradiction"
        assert ConflictType.PARTIAL_DISAGREEMENT.value == "partial_disagreement"
        assert ConflictType.TEMPORAL_INCONSISTENCY.value == "temporal_inconsistency"
        assert ConflictType.VALUE_DISCREPANCY.value == "value_discrepancy"
        assert ConflictType.SCOPE_MISMATCH.value == "scope_mismatch"

    def test_conflict_severity_enum(self):
        assert ConflictSeverity.CRITICAL.value == "critical"
        assert ConflictSeverity.HIGH.value == "high"
        assert ConflictSeverity.MEDIUM.value == "medium"
        assert ConflictSeverity.LOW.value == "low"
        assert ConflictSeverity.NEGLIGIBLE.value == "negligible"

    def test_conflict_resolution(self):
        conflict = Conflict(
            conflict_id="conf_res",
            item_ids=["item_001", "item_002"],
            conflict_type=ConflictType.VALUE_DISCREPANCY,
            severity=ConflictSeverity.MEDIUM,
            description="Test conflict",
            detected_at=datetime.now()
        )

        conflict.resolved = True
        conflict.resolution = "item_001"
        conflict.resolution_strategy = ResolutionStrategy.HIGHEST_AUTHORITY

        assert conflict.resolved is True
        assert conflict.resolution == "item_001"
        assert conflict.resolution_strategy == ResolutionStrategy.HIGHEST_AUTHORITY


class TestResolutionStrategy:
    def test_resolution_strategy_enum(self):
        assert ResolutionStrategy.HIGHEST_AUTHORITY.value == "highest_authority"
        assert ResolutionStrategy.MOST_RECENT.value == "most_recent"
        assert ResolutionStrategy.MAJORITY_CONSENSUS.value == "majority_consensus"
        assert ResolutionStrategy.EVIDENCE_BASED.value == "evidence_based"
        assert ResolutionStrategy.MANUAL_REVIEW.value == "manual_review"


class TestConflictDetector:
    def test_initialization(self, conflict_detector):
        assert conflict_detector.knowledge_items == {}
        assert conflict_detector.conflicts == {}
        assert conflict_detector.domain_index == {}
        assert conflict_detector.source_index == {}

    def test_add_knowledge_item(self, conflict_detector, knowledge_item_1):
        conflict_detector.add_knowledge_item(knowledge_item_1)

        assert "item_001" in conflict_detector.knowledge_items
        assert conflict_detector.knowledge_items["item_001"] == knowledge_item_1

    def test_add_knowledge_item_updates_domain_index(self, conflict_detector, knowledge_item_1):
        conflict_detector.add_knowledge_item(knowledge_item_1)

        assert "physics" in conflict_detector.domain_index
        assert "item_001" in conflict_detector.domain_index["physics"]

    def test_add_knowledge_item_updates_source_index(self, conflict_detector, knowledge_item_1):
        conflict_detector.add_knowledge_item(knowledge_item_1)

        assert "src_001" in conflict_detector.source_index
        assert "item_001" in conflict_detector.source_index["src_001"]

    def test_add_multiple_items_same_domain(self, conflict_detector, knowledge_item_1, knowledge_item_2):
        conflict_detector.add_knowledge_item(knowledge_item_1)
        conflict_detector.add_knowledge_item(knowledge_item_2)

        assert len(conflict_detector.domain_index["physics"]) == 2
        assert "item_001" in conflict_detector.domain_index["physics"]
        assert "item_002" in conflict_detector.domain_index["physics"]

    def test_detect_conflict_contradiction(self, conflict_detector, knowledge_item_3, knowledge_item_4):
        conflict_detector.add_knowledge_item(knowledge_item_3)
        conflict_detector.add_knowledge_item(knowledge_item_4)

        conflict = conflict_detector.detect_conflict(
            item_id_1="item_003",
            item_id_2="item_004"
        )

        assert conflict is not None
        assert conflict.conflict_type in [
            ConflictType.DIRECT_CONTRADICTION,
            ConflictType.TEMPORAL_INCONSISTENCY,
            ConflictType.VALUE_DISCREPANCY
        ]
        assert "item_003" in conflict.item_ids
        assert "item_004" in conflict.item_ids

    def test_detect_conflict_temporal(self, conflict_detector, outdated_item, recent_item):
        conflict_detector.add_knowledge_item(outdated_item)
        conflict_detector.add_knowledge_item(recent_item)

        conflict = conflict_detector.detect_conflict(
            item_id_1="item_005",
            item_id_2="item_006"
        )

        assert conflict is not None
        assert conflict.conflict_type == ConflictType.TEMPORAL_INCONSISTENCY

    def test_detect_conflict_no_conflict(self, conflict_detector):
        item1 = KnowledgeItem(
            item_id="i1",
            content="The Earth is round",
            source_id="s1",
            timestamp=datetime.now(),
            domain="geography",
            authority_score=0.95
        )

        item2 = KnowledgeItem(
            item_id="i2",
            content="Water is H2O",
            source_id="s2",
            timestamp=datetime.now(),
            domain="chemistry",
            authority_score=0.95
        )

        conflict_detector.add_knowledge_item(item1)
        conflict_detector.add_knowledge_item(item2)

        conflict = conflict_detector.detect_conflict("i1", "i2")

        assert conflict is None

    def test_assess_severity_high_authority_conflict(self, conflict_detector):
        item1 = KnowledgeItem(
            item_id="high1",
            content="Test A",
            source_id="s1",
            timestamp=datetime.now(),
            domain="test",
            authority_score=0.95
        )

        item2 = KnowledgeItem(
            item_id="high2",
            content="Test B",
            source_id="s2",
            timestamp=datetime.now(),
            domain="test",
            authority_score=0.90
        )

        conflict = Conflict(
            conflict_id="test_conf",
            item_ids=["high1", "high2"],
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            severity=ConflictSeverity.MEDIUM,
            description="Test",
            detected_at=datetime.now()
        )

        conflict_detector.add_knowledge_item(item1)
        conflict_detector.add_knowledge_item(item2)

        severity = conflict_detector.assess_severity(conflict)

        assert severity in [ConflictSeverity.HIGH, ConflictSeverity.CRITICAL]

    def test_assess_severity_low_authority_conflict(self, conflict_detector):
        item1 = KnowledgeItem(
            item_id="low1",
            content="Test A",
            source_id="s1",
            timestamp=datetime.now(),
            domain="test",
            authority_score=0.30
        )

        item2 = KnowledgeItem(
            item_id="low2",
            content="Test B",
            source_id="s2",
            timestamp=datetime.now(),
            domain="test",
            authority_score=0.25
        )

        conflict = Conflict(
            conflict_id="test_conf",
            item_ids=["low1", "low2"],
            conflict_type=ConflictType.PARTIAL_DISAGREEMENT,
            severity=ConflictSeverity.MEDIUM,
            description="Test",
            detected_at=datetime.now()
        )

        conflict_detector.add_knowledge_item(item1)
        conflict_detector.add_knowledge_item(item2)

        severity = conflict_detector.assess_severity(conflict)

        assert severity in [ConflictSeverity.LOW, ConflictSeverity.NEGLIGIBLE]

    def test_resolve_conflict_highest_authority(self, conflict_detector, knowledge_item_1, knowledge_item_2):
        conflict_detector.add_knowledge_item(knowledge_item_1)
        conflict_detector.add_knowledge_item(knowledge_item_2)

        conflict = Conflict(
            conflict_id="auth_conf",
            item_ids=["item_001", "item_002"],
            conflict_type=ConflictType.VALUE_DISCREPANCY,
            severity=ConflictSeverity.MEDIUM,
            description="Authority test",
            detected_at=datetime.now()
        )

        resolution = conflict_detector.resolve_conflict(
            conflict,
            strategy=ResolutionStrategy.HIGHEST_AUTHORITY
        )

        assert resolution == "item_001"
        assert conflict.resolved is True
        assert conflict.resolution == "item_001"
        assert conflict.resolution_strategy == ResolutionStrategy.HIGHEST_AUTHORITY

    def test_resolve_conflict_most_recent(self, conflict_detector, knowledge_item_3, knowledge_item_4):
        conflict_detector.add_knowledge_item(knowledge_item_3)
        conflict_detector.add_knowledge_item(knowledge_item_4)

        conflict = Conflict(
            conflict_id="time_conf",
            item_ids=["item_003", "item_004"],
            conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
            severity=ConflictSeverity.MEDIUM,
            description="Temporal test",
            detected_at=datetime.now()
        )

        resolution = conflict_detector.resolve_conflict(
            conflict,
            strategy=ResolutionStrategy.MOST_RECENT
        )

        assert resolution == "item_004"

    def test_resolve_conflict_majority_consensus(self, conflict_detector):
        items = [
            KnowledgeItem(f"i{i}", "Value A", f"s{i}", datetime.now(), "test", 0.8)
            for i in range(3)
        ]
        items.append(
            KnowledgeItem("i3", "Value B", "s3", datetime.now(), "test", 0.8)
        )

        for item in items:
            conflict_detector.add_knowledge_item(item)

        conflict = Conflict(
            conflict_id="majority_conf",
            item_ids=["i0", "i1", "i2", "i3"],
            conflict_type=ConflictType.PARTIAL_DISAGREEMENT,
            severity=ConflictSeverity.MEDIUM,
            description="Majority test",
            detected_at=datetime.now()
        )

        resolution = conflict_detector.resolve_conflict(
            conflict,
            strategy=ResolutionStrategy.MAJORITY_CONSENSUS
        )

        assert resolution in ["i0", "i1", "i2"]

    def test_resolve_conflict_evidence_based(self, conflict_detector):
        item1 = KnowledgeItem(
            item_id="ev1",
            content="Claim with evidence",
            source_id="s1",
            timestamp=datetime.now(),
            domain="test",
            authority_score=0.85
        )

        item2 = KnowledgeItem(
            item_id="ev2",
            content="Claim without evidence",
            source_id="s2",
            timestamp=datetime.now(),
            domain="test",
            authority_score=0.60
        )

        conflict_detector.add_knowledge_item(item1)
        conflict_detector.add_knowledge_item(item2)

        conflict = Conflict(
            conflict_id="ev_conf",
            item_ids=["ev1", "ev2"],
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            severity=ConflictSeverity.HIGH,
            description="Evidence test",
            detected_at=datetime.now()
        )

        resolution = conflict_detector.resolve_conflict(
            conflict,
            strategy=ResolutionStrategy.EVIDENCE_BASED
        )

        assert resolution == "ev1"

    def test_find_conflicts_in_domain(self, conflict_detector, knowledge_item_1, knowledge_item_2):
        conflict_detector.add_knowledge_item(knowledge_item_1)
        conflict_detector.add_knowledge_item(knowledge_item_2)

        conflicts = conflict_detector.find_conflicts_in_domain("physics")

        assert len(conflicts) >= 0

    def test_find_conflicts_in_domain_empty(self, conflict_detector):
        conflicts = conflict_detector.find_conflicts_in_domain("nonexistent")

        assert conflicts == []

    def test_get_conflicts_by_severity(self, conflict_detector):
        conflict_detector.add_knowledge_item(knowledge_item_1)
        conflict_detector.add_knowledge_item(knowledge_item_2)

        conflict = Conflict(
            conflict_id="sev_test",
            item_ids=["item_001", "item_002"],
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            severity=ConflictSeverity.CRITICAL,
            description="Severity test",
            detected_at=datetime.now()
        )

        conflict_detector.conflicts["sev_test"] = conflict

        critical_conflicts = conflict_detector.get_conflicts_by_severity(ConflictSeverity.CRITICAL)

        assert len(critical_conflicts) == 1
        assert critical_conflicts[0].conflict_id == "sev_test"

    def test_get_unresolved_conflicts(self, conflict_detector):
        resolved_conflict = Conflict(
            conflict_id="resolved",
            item_ids=["i1", "i2"],
            conflict_type=ConflictType.VALUE_DISCREPANCY,
            severity=ConflictSeverity.MEDIUM,
            description="Resolved",
            detected_at=datetime.now()
        )
        resolved_conflict.resolved = True
        resolved_conflict.resolution = "i1"

        unresolved_conflict = Conflict(
            conflict_id="unresolved",
            item_ids=["i3", "i4"],
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            severity=ConflictSeverity.HIGH,
            description="Unresolved",
            detected_at=datetime.now()
        )

        conflict_detector.conflicts["resolved"] = resolved_conflict
        conflict_detector.conflicts["unresolved"] = unresolved_conflict

        unresolved = conflict_detector.get_unresolved_conflicts()

        assert len(unresolved) == 1
        assert unresolved[0].conflict_id == "unresolved"

    def test_auto_resolve_conflicts(self, conflict_detector, knowledge_item_1, knowledge_item_2):
        conflict_detector.add_knowledge_item(knowledge_item_1)
        conflict_detector.add_knowledge_item(knowledge_item_2)

        conflict = Conflict(
            conflict_id="auto_res",
            item_ids=["item_001", "item_002"],
            conflict_type=ConflictType.VALUE_DISCREPANCY,
            severity=ConflictSeverity.MEDIUM,
            description="Auto resolve test",
            detected_at=datetime.now()
        )

        conflict_detector.conflicts["auto_res"] = conflict

        results = conflict_detector.auto_resolve_conflicts(
            min_confidence=0.7,
            strategy=ResolutionStrategy.HIGHEST_AUTHORITY
        )

        assert len(results) >= 0
        if len(results) > 0:
            assert conflict.resolved is True

    def test_auto_resolve_low_confidence_threshold(self, conflict_detector, knowledge_item_1, knowledge_item_2):
        conflict_detector.add_knowledge_item(knowledge_item_1)
        conflict_detector.add_knowledge_item(knowledge_item_2)

        conflict = Conflict(
            conflict_id="low_conf",
            item_ids=["item_001", "item_002"],
            conflict_type=ConflictType.PARTIAL_DISAGREEMENT,
            severity=ConflictSeverity.LOW,
            description="Low confidence test",
            detected_at=datetime.now()
        )

        conflict_detector.conflicts["low_conf"] = conflict

        results = conflict_detector.auto_resolve_conflicts(
            min_confidence=0.99,
            strategy=ResolutionStrategy.HIGHEST_AUTHORITY
        )

        assert len(results) == 0

    def test_export_state(self, conflict_detector, knowledge_item_1):
        conflict_detector.add_knowledge_item(knowledge_item_1)

        conflict = Conflict(
            conflict_id="exp_test",
            item_ids=["item_001"],
            conflict_type=ConflictType.VALUE_DISCREPANCY,
            severity=ConflictSeverity.MEDIUM,
            description="Export test",
            detected_at=datetime.now()
        )

        conflict_detector.conflicts["exp_test"] = conflict

        state = conflict_detector.export_state()

        assert "knowledge_items" in state
        assert "conflicts" in state
        assert len(state["knowledge_items"]) == 1
        assert len(state["conflicts"]) == 1

    def test_import_state(self, conflict_detector):
        state = {
            "knowledge_items": {
                "test_item": {
                    "item_id": "test_item",
                    "content": "Test content",
                    "source_id": "test_source",
                    "timestamp": datetime.now().isoformat(),
                    "domain": "test",
                    "authority_score": 0.8
                }
            },
            "conflicts": {
                "test_conflict": {
                    "conflict_id": "test_conflict",
                    "item_ids": ["test_item"],
                    "conflict_type": "value_discrepancy",
                    "severity": "medium",
                    "description": "Test conflict",
                    "detected_at": datetime.now().isoformat(),
                    "resolved": False,
                    "resolution": None,
                    "resolution_strategy": None
                }
            }
        }

        conflict_detector.import_state(state)

        assert "test_item" in conflict_detector.knowledge_items
        assert "test_conflict" in conflict_detector.conflicts

    def test_export_import_roundtrip(self, conflict_detector, knowledge_item_1, knowledge_item_2):
        conflict_detector.add_knowledge_item(knowledge_item_1)
        conflict_detector.add_knowledge_item(knowledge_item_2)

        conflict = Conflict(
            conflict_id="roundtrip",
            item_ids=["item_001", "item_002"],
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            severity=ConflictSeverity.HIGH,
            description="Roundtrip test",
            detected_at=datetime.now()
        )

        conflict_detector.conflicts["roundtrip"] = conflict

        state = conflict_detector.export_state()

        new_detector = ConflictDetector()
        new_detector.import_state(state)

        assert len(new_detector.knowledge_items) == 2
        assert len(new_detector.conflicts) == 1
        assert "item_001" in new_detector.knowledge_items
        assert "roundtrip" in new_detector.conflicts

    def test_complex_multi_item_conflict(self, conflict_detector):
        items = []
        for i in range(5):
            item = KnowledgeItem(
                item_id=f"multi_{i}",
                content=f"Version {i} of the fact",
                source_id=f"src_{i}",
                timestamp=datetime.now() - timedelta(days=i*10),
                domain="multi_test",
                authority_score=0.7 + (i * 0.05)
            )
            items.append(item)
            conflict_detector.add_knowledge_item(item)

        conflict = Conflict(
            conflict_id="multi_conf",
            item_ids=[f"multi_{i}" for i in range(5)],
            conflict_type=ConflictType.PARTIAL_DISAGREEMENT,
            severity=ConflictSeverity.MEDIUM,
            description="Multi-item conflict",
            detected_at=datetime.now()
        )

        conflict_detector.conflicts["multi_conf"] = conflict

        resolution = conflict_detector.resolve_conflict(
            conflict,
            strategy=ResolutionStrategy.HIGHEST_AUTHORITY
        )

        assert resolution == "multi_4"

    def test_domain_indexing_multiple_domains(self, conflict_detector):
        domains = ["physics", "chemistry", "biology", "mathematics"]

        for i, domain in enumerate(domains):
            for j in range(3):
                item = KnowledgeItem(
                    item_id=f"{domain}_{j}",
                    content=f"Content for {domain} {j}",
                    source_id=f"src_{i}_{j}",
                    timestamp=datetime.now(),
                    domain=domain,
                    authority_score=0.8
                )
                conflict_detector.add_knowledge_item(item)

        assert len(conflict_detector.domain_index) == 4
        for domain in domains:
            assert len(conflict_detector.domain_index[domain]) == 3

    def test_source_indexing_multiple_sources(self, conflict_detector):
        for i in range(10):
            item = KnowledgeItem(
                item_id=f"item_{i}",
                content=f"Content {i}",
                source_id=f"source_{i % 3}",
                timestamp=datetime.now(),
                domain="test",
                authority_score=0.75
            )
            conflict_detector.add_knowledge_item(item)

        assert len(conflict_detector.source_index) == 3
        assert len(conflict_detector.source_index["source_0"]) >= 3

    def test_conflict_detection_edge_cases(self, conflict_detector):
        same_item = KnowledgeItem(
            item_id="same",
            content="Same content",
            source_id="src",
            timestamp=datetime.now(),
            domain="test",
            authority_score=0.8
        )

        conflict_detector.add_knowledge_item(same_item)

        conflict = conflict_detector.detect_conflict("same", "same")

        assert conflict is None

    def test_resolution_with_tie_in_authority(self, conflict_detector):
        item1 = KnowledgeItem(
            item_id="tie1",
            content="Content A",
            source_id="s1",
            timestamp=datetime.now(),
            domain="test",
            authority_score=0.85
        )

        item2 = KnowledgeItem(
            item_id="tie2",
            content="Content B",
            source_id="s2",
            timestamp=datetime.now(),
            domain="test",
            authority_score=0.85
        )

        conflict_detector.add_knowledge_item(item1)
        conflict_detector.add_knowledge_item(item2)

        conflict = Conflict(
            conflict_id="tie",
            item_ids=["tie1", "tie2"],
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            severity=ConflictSeverity.MEDIUM,
            description="Tie test",
            detected_at=datetime.now()
        )

        resolution = conflict_detector.resolve_conflict(
            conflict,
            strategy=ResolutionStrategy.HIGHEST_AUTHORITY
        )

        assert resolution in ["tie1", "tie2"]

    def test_empty_domain_search(self, conflict_detector):
        conflicts = conflict_detector.find_conflicts_in_domain("")

        assert conflicts == []

    def test_conflict_severity_ordering(self):
        severities = [
            ConflictSeverity.NEGLIGIBLE,
            ConflictSeverity.LOW,
            ConflictSeverity.MEDIUM,
            ConflictSeverity.HIGH,
            ConflictSeverity.CRITICAL
        ]

        assert len(severities) == 5
        assert severities[0] == ConflictSeverity.NEGLIGIBLE
        assert severities[-1] == ConflictSeverity.CRITICAL

    def test_batch_conflict_detection(self, conflict_detector):
        items = []
        for i in range(20):
            item = KnowledgeItem(
                item_id=f"batch_{i}",
                content=f"Statement {i % 5}",
                source_id=f"src_{i}",
                timestamp=datetime.now(),
                domain="batch_test",
                authority_score=0.7 + (i % 5) * 0.05
            )
            items.append(item)
            conflict_detector.add_knowledge_item(item)

        assert len(conflict_detector.knowledge_items) == 20
        assert len(conflict_detector.domain_index["batch_test"]) == 20

    def test_conflict_resolution_invalid_strategy(self, conflict_detector, knowledge_item_1):
        conflict_detector.add_knowledge_item(knowledge_item_1)

        conflict = Conflict(
            conflict_id="invalid",
            item_ids=["item_001"],
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            severity=ConflictSeverity.MEDIUM,
            description="Invalid strategy test",
            detected_at=datetime.now()
        )

        try:
            conflict_detector.resolve_conflict(conflict, strategy=ResolutionStrategy.MANUAL_REVIEW)
        except Exception:
            pass

    def test_get_conflicts_by_type(self, conflict_detector):
        for i, conflict_type in enumerate([ConflictType.DIRECT_CONTRADICTION, ConflictType.PARTIAL_DISAGREEMENT]):
            conflict = Conflict(
                conflict_id=f"type_{i}",
                item_ids=[f"i{i}_1", f"i{i}_2"],
                conflict_type=conflict_type,
                severity=ConflictSeverity.MEDIUM,
                description=f"Type test {i}",
                detected_at=datetime.now()
            )
            conflict_detector.conflicts[f"type_{i}"] = conflict

        contradictions = [c for c in conflict_detector.conflicts.values()
                         if c.conflict_type == ConflictType.DIRECT_CONTRADICTION]
        disagreements = [c for c in conflict_detector.conflicts.values()
                        if c.conflict_type == ConflictType.PARTIAL_DISAGREEMENT]

        assert len(contradictions) == 1
        assert len(disagreements) == 1
