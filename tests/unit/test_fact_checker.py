import pytest
from datetime import datetime, timedelta
from fractal_agent.knowledge.fact_checking.fact_checker import (
    FactChecker,
    Fact,
    FactStatus,
    Source,
    SourceType,
    AuthorityScore
)
import json
import tempfile
import os


@pytest.fixture
def fact_checker():
    return FactChecker()


@pytest.fixture
def primary_source():
    return Source(
        source_id="src_001",
        name="Academic Journal",
        source_type=SourceType.PRIMARY,
        reliability_score=0.95,
        expertise_domains=["machine_learning", "ai"],
        publication_date=datetime.now(),
        citations_count=150
    )


@pytest.fixture
def secondary_source():
    return Source(
        source_id="src_002",
        name="Tech Blog",
        source_type=SourceType.SECONDARY,
        reliability_score=0.70,
        expertise_domains=["ai"],
        publication_date=datetime.now() - timedelta(days=30),
        citations_count=10
    )


@pytest.fixture
def outdated_source():
    return Source(
        source_id="src_003",
        name="Old Paper",
        source_type=SourceType.PRIMARY,
        reliability_score=0.90,
        expertise_domains=["ai"],
        publication_date=datetime.now() - timedelta(days=400),
        citations_count=200
    )


@pytest.fixture
def ai_source():
    return Source(
        source_id="src_004",
        name="AI Model",
        source_type=SourceType.AI_GENERATED,
        reliability_score=0.60,
        expertise_domains=["general"],
        publication_date=datetime.now(),
        citations_count=0
    )


@pytest.fixture
def sample_fact():
    return Fact(
        fact_id="fact_001",
        statement="Neural networks require gradient descent for training",
        domain="machine_learning",
        timestamp=datetime.now()
    )


class TestSource:
    def test_source_creation(self, primary_source):
        assert primary_source.source_id == "src_001"
        assert primary_source.name == "Academic Journal"
        assert primary_source.source_type == SourceType.PRIMARY
        assert primary_source.reliability_score == 0.95
        assert "machine_learning" in primary_source.expertise_domains
        assert primary_source.citations_count == 150

    def test_source_type_enum(self):
        assert SourceType.PRIMARY.value == "primary"
        assert SourceType.SECONDARY.value == "secondary"
        assert SourceType.EXPERT.value == "expert"
        assert SourceType.AI_GENERATED.value == "ai_generated"

    def test_source_with_minimal_data(self):
        source = Source(
            source_id="minimal",
            name="Minimal Source",
            source_type=SourceType.SECONDARY,
            reliability_score=0.5,
            expertise_domains=[],
            publication_date=datetime.now(),
            citations_count=0
        )
        assert source.source_id == "minimal"
        assert source.expertise_domains == []
        assert source.citations_count == 0


class TestFact:
    def test_fact_creation(self, sample_fact):
        assert sample_fact.fact_id == "fact_001"
        assert "Neural networks" in sample_fact.statement
        assert sample_fact.domain == "machine_learning"
        assert sample_fact.status == FactStatus.UNVERIFIED
        assert sample_fact.sources == []
        assert sample_fact.confidence_score == 0.0

    def test_fact_status_enum(self):
        assert FactStatus.VERIFIED.value == "verified"
        assert FactStatus.DISPUTED.value == "disputed"
        assert FactStatus.UNVERIFIED.value == "unverified"
        assert FactStatus.OUTDATED.value == "outdated"
        assert FactStatus.CONFLICTING.value == "conflicting"

    def test_fact_with_sources(self, sample_fact, primary_source):
        sample_fact.sources.append(primary_source.source_id)
        sample_fact.confidence_score = 0.85
        sample_fact.status = FactStatus.VERIFIED

        assert len(sample_fact.sources) == 1
        assert sample_fact.sources[0] == "src_001"
        assert sample_fact.confidence_score == 0.85
        assert sample_fact.status == FactStatus.VERIFIED


class TestAuthorityScore:
    def test_authority_score_creation(self):
        score = AuthorityScore(
            source_id="src_001",
            composite_score=0.88,
            expertise_score=0.90,
            reliability_score=0.95,
            recency_score=0.80,
            citation_score=0.85
        )
        assert score.source_id == "src_001"
        assert score.composite_score == 0.88
        assert score.expertise_score == 0.90

    def test_authority_score_bounds(self):
        score = AuthorityScore(
            source_id="test",
            composite_score=1.0,
            expertise_score=0.0,
            reliability_score=0.5,
            recency_score=1.0,
            citation_score=0.0
        )
        assert 0.0 <= score.composite_score <= 1.0
        assert 0.0 <= score.expertise_score <= 1.0


class TestFactChecker:
    def test_initialization(self, fact_checker):
        assert fact_checker.facts == {}
        assert fact_checker.sources == {}
        assert fact_checker.authority_scores == {}

    def test_add_source(self, fact_checker, primary_source):
        fact_checker.add_source(primary_source)
        assert "src_001" in fact_checker.sources
        assert fact_checker.sources["src_001"] == primary_source

    def test_add_multiple_sources(self, fact_checker, primary_source, secondary_source):
        fact_checker.add_source(primary_source)
        fact_checker.add_source(secondary_source)
        assert len(fact_checker.sources) == 2
        assert "src_001" in fact_checker.sources
        assert "src_002" in fact_checker.sources

    def test_add_fact(self, fact_checker, sample_fact):
        fact_checker.add_fact(sample_fact)
        assert "fact_001" in fact_checker.facts
        assert fact_checker.facts["fact_001"] == sample_fact

    def test_calculate_authority_score_primary_source(self, fact_checker, primary_source):
        fact_checker.add_source(primary_source)
        score = fact_checker.calculate_authority_score(
            source_id="src_001",
            domain="machine_learning"
        )

        assert score.source_id == "src_001"
        assert score.composite_score > 0.8
        assert score.expertise_score == 1.0
        assert score.reliability_score == 0.95
        assert score.recency_score >= 0.9

    def test_calculate_authority_score_mismatched_domain(self, fact_checker, primary_source):
        fact_checker.add_source(primary_source)
        score = fact_checker.calculate_authority_score(
            source_id="src_001",
            domain="biology"
        )

        assert score.expertise_score == 0.0
        assert score.composite_score < 0.8

    def test_calculate_authority_score_outdated_source(self, fact_checker, outdated_source):
        fact_checker.add_source(outdated_source)
        score = fact_checker.calculate_authority_score(
            source_id="src_003",
            domain="ai"
        )

        assert score.recency_score < 0.5

    def test_calculate_authority_score_ai_generated(self, fact_checker, ai_source):
        fact_checker.add_source(ai_source)
        score = fact_checker.calculate_authority_score(
            source_id="src_004",
            domain="general"
        )

        assert score.reliability_score == 0.60
        assert score.citation_score == 0.0

    def test_calculate_authority_score_nonexistent_source(self, fact_checker):
        with pytest.raises(KeyError):
            fact_checker.calculate_authority_score(
                source_id="nonexistent",
                domain="test"
            )

    def test_validate_fact_single_source(self, fact_checker, sample_fact, primary_source):
        fact_checker.add_source(primary_source)
        fact_checker.add_fact(sample_fact)

        result = fact_checker.validate_fact(
            fact_id="fact_001",
            supporting_sources=["src_001"]
        )

        assert result["fact_id"] == "fact_001"
        assert result["status"] == FactStatus.VERIFIED
        assert result["confidence"] > 0.8
        assert len(result["supporting_sources"]) == 1
        assert "src_001" in result["authority_scores"]

    def test_validate_fact_multiple_sources(self, fact_checker, sample_fact, primary_source, secondary_source):
        fact_checker.add_source(primary_source)
        fact_checker.add_source(secondary_source)
        fact_checker.add_fact(sample_fact)

        result = fact_checker.validate_fact(
            fact_id="fact_001",
            supporting_sources=["src_001", "src_002"]
        )

        assert result["status"] == FactStatus.VERIFIED
        assert len(result["supporting_sources"]) == 2
        assert len(result["authority_scores"]) == 2

    def test_validate_fact_no_sources(self, fact_checker, sample_fact):
        fact_checker.add_fact(sample_fact)

        result = fact_checker.validate_fact(
            fact_id="fact_001",
            supporting_sources=[]
        )

        assert result["status"] == FactStatus.UNVERIFIED
        assert result["confidence"] == 0.0

    def test_validate_fact_low_quality_sources(self, fact_checker, sample_fact, ai_source):
        fact_checker.add_source(ai_source)
        fact_checker.add_fact(sample_fact)

        result = fact_checker.validate_fact(
            fact_id="fact_001",
            supporting_sources=["src_004"]
        )

        assert result["confidence"] < 0.7

    def test_check_currency_recent_source(self, fact_checker, primary_source):
        fact_checker.add_source(primary_source)

        is_current = fact_checker.check_currency(
            source_id="src_001",
            max_age_days=365
        )

        assert is_current is True

    def test_check_currency_outdated_source(self, fact_checker, outdated_source):
        fact_checker.add_source(outdated_source)

        is_current = fact_checker.check_currency(
            source_id="src_003",
            max_age_days=365
        )

        assert is_current is False

    def test_check_currency_custom_threshold(self, fact_checker, secondary_source):
        fact_checker.add_source(secondary_source)

        is_current_strict = fact_checker.check_currency(
            source_id="src_002",
            max_age_days=7
        )
        is_current_lenient = fact_checker.check_currency(
            source_id="src_002",
            max_age_days=60
        )

        assert is_current_strict is False
        assert is_current_lenient is True

    def test_get_outdated_sources(self, fact_checker, primary_source, outdated_source):
        fact_checker.add_source(primary_source)
        fact_checker.add_source(outdated_source)

        outdated = fact_checker.get_outdated_sources(max_age_days=365)

        assert len(outdated) == 1
        assert "src_003" in outdated
        assert "src_001" not in outdated

    def test_get_outdated_sources_empty(self, fact_checker, primary_source):
        fact_checker.add_source(primary_source)

        outdated = fact_checker.get_outdated_sources(max_age_days=365)

        assert len(outdated) == 0

    def test_export_state(self, fact_checker, sample_fact, primary_source):
        fact_checker.add_source(primary_source)
        fact_checker.add_fact(sample_fact)
        fact_checker.validate_fact("fact_001", ["src_001"])

        state = fact_checker.export_state()

        assert "facts" in state
        assert "sources" in state
        assert "authority_scores" in state
        assert len(state["facts"]) == 1
        assert len(state["sources"]) == 1

    def test_export_import_roundtrip(self, fact_checker, sample_fact, primary_source, secondary_source):
        fact_checker.add_source(primary_source)
        fact_checker.add_source(secondary_source)
        fact_checker.add_fact(sample_fact)
        fact_checker.validate_fact("fact_001", ["src_001", "src_002"])

        state = fact_checker.export_state()

        new_checker = FactChecker()
        new_checker.import_state(state)

        assert len(new_checker.facts) == len(fact_checker.facts)
        assert len(new_checker.sources) == len(fact_checker.sources)
        assert "fact_001" in new_checker.facts
        assert "src_001" in new_checker.sources
        assert "src_002" in new_checker.sources

    def test_export_to_file(self, fact_checker, sample_fact, primary_source):
        fact_checker.add_source(primary_source)
        fact_checker.add_fact(sample_fact)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            state = fact_checker.export_state()
            with open(temp_path, 'w') as f:
                json.dump(state, f)

            assert os.path.exists(temp_path)

            with open(temp_path, 'r') as f:
                loaded_state = json.load(f)

            assert "facts" in loaded_state
            assert "sources" in loaded_state
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_import_empty_state(self, fact_checker):
        empty_state = {
            "facts": {},
            "sources": {},
            "authority_scores": {}
        }

        fact_checker.import_state(empty_state)

        assert len(fact_checker.facts) == 0
        assert len(fact_checker.sources) == 0

    def test_complex_validation_workflow(self, fact_checker):
        source1 = Source(
            source_id="s1",
            name="Expert",
            source_type=SourceType.EXPERT,
            reliability_score=0.98,
            expertise_domains=["physics"],
            publication_date=datetime.now(),
            citations_count=500
        )

        source2 = Source(
            source_id="s2",
            name="Journal",
            source_type=SourceType.PRIMARY,
            reliability_score=0.95,
            expertise_domains=["physics"],
            publication_date=datetime.now() - timedelta(days=10),
            citations_count=300
        )

        fact = Fact(
            fact_id="f1",
            statement="E=mc²",
            domain="physics",
            timestamp=datetime.now()
        )

        fact_checker.add_source(source1)
        fact_checker.add_source(source2)
        fact_checker.add_fact(fact)

        result = fact_checker.validate_fact("f1", ["s1", "s2"])

        assert result["status"] == FactStatus.VERIFIED
        assert result["confidence"] > 0.9
        assert len(result["authority_scores"]) == 2

    def test_edge_case_empty_statement(self):
        fact = Fact(
            fact_id="empty",
            statement="",
            domain="test",
            timestamp=datetime.now()
        )

        assert fact.statement == ""
        assert fact.fact_id == "empty"

    def test_edge_case_very_long_statement(self):
        long_statement = "A" * 10000
        fact = Fact(
            fact_id="long",
            statement=long_statement,
            domain="test",
            timestamp=datetime.now()
        )

        assert len(fact.statement) == 10000

    def test_edge_case_zero_reliability(self):
        source = Source(
            source_id="unreliable",
            name="Unreliable",
            source_type=SourceType.SECONDARY,
            reliability_score=0.0,
            expertise_domains=["test"],
            publication_date=datetime.now(),
            citations_count=0
        )

        assert source.reliability_score == 0.0

    def test_edge_case_max_reliability(self):
        source = Source(
            source_id="perfect",
            name="Perfect",
            source_type=SourceType.PRIMARY,
            reliability_score=1.0,
            expertise_domains=["test"],
            publication_date=datetime.now(),
            citations_count=1000
        )

        assert source.reliability_score == 1.0

    def test_multiple_domains_expertise(self):
        source = Source(
            source_id="multi",
            name="Multi-domain Expert",
            source_type=SourceType.EXPERT,
            reliability_score=0.92,
            expertise_domains=["ai", "ml", "nlp", "cv"],
            publication_date=datetime.now(),
            citations_count=800
        )

        assert len(source.expertise_domains) == 4
        assert "ai" in source.expertise_domains
        assert "cv" in source.expertise_domains

    def test_fact_metadata_preservation(self, fact_checker, sample_fact):
        original_timestamp = sample_fact.timestamp
        fact_checker.add_fact(sample_fact)

        retrieved_fact = fact_checker.facts["fact_001"]
        assert retrieved_fact.timestamp == original_timestamp
        assert retrieved_fact.domain == sample_fact.domain

    def test_concurrent_fact_validation(self, fact_checker, primary_source):
        fact_checker.add_source(primary_source)

        facts = [
            Fact(f"fact_{i}", f"Statement {i}", "machine_learning", datetime.now())
            for i in range(10)
        ]

        for fact in facts:
            fact_checker.add_fact(fact)

        results = []
        for fact in facts:
            result = fact_checker.validate_fact(fact.fact_id, ["src_001"])
            results.append(result)

        assert len(results) == 10
        assert all(r["status"] == FactStatus.VERIFIED for r in results)

    def test_authority_score_caching(self, fact_checker, primary_source):
        fact_checker.add_source(primary_source)

        score1 = fact_checker.calculate_authority_score("src_001", "machine_learning")
        score2 = fact_checker.calculate_authority_score("src_001", "machine_learning")

        key = "src_001:machine_learning"
        assert key in fact_checker.authority_scores
        assert score1.composite_score == score2.composite_score

    def test_validate_with_conflicting_authority(self, fact_checker):
        high_authority = Source(
            source_id="high",
            name="High Authority",
            source_type=SourceType.PRIMARY,
            reliability_score=0.95,
            expertise_domains=["test"],
            publication_date=datetime.now(),
            citations_count=1000
        )

        low_authority = Source(
            source_id="low",
            name="Low Authority",
            source_type=SourceType.AI_GENERATED,
            reliability_score=0.50,
            expertise_domains=[],
            publication_date=datetime.now() - timedelta(days=100),
            citations_count=0
        )

        fact = Fact("test_fact", "Test statement", "test", datetime.now())

        fact_checker.add_source(high_authority)
        fact_checker.add_source(low_authority)
        fact_checker.add_fact(fact)

        result = fact_checker.validate_fact("test_fact", ["high", "low"])

        assert result["confidence"] > 0.7
        assert "high" in result["authority_scores"]
        assert "low" in result["authority_scores"]
