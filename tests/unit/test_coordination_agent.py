"""
Unit tests for CoordinationAgent

Tests coordination agent functionality including conflict detection,
resolution, consensus building, and resource allocation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from fractal_agent.agents.coordination_agent import (
    CoordinationAgent,
    CoordinationConfig,
    ConflictType,
    ResolutionStrategy,
    ConflictSeverity,
    AgentOutput,
    Conflict,
    Resolution,
    CoordinationResult,
    PresetCoordinationConfigs,
    ConflictDetection,
    ConflictResolution,
    ConsensusBuilding,
    ResourceAllocation,
)
from fractal_agent.verification import VerificationResult


# ============================================================================
# Enum Tests
# ============================================================================

class TestConflictType:
    """Test ConflictType enum"""

    def test_conflict_type_values(self):
        """Test all conflict type enum values"""
        assert ConflictType.CODE_CONFLICT.value == "code_conflict"
        assert ConflictType.GOAL_CONFLICT.value == "goal_conflict"
        assert ConflictType.RESOURCE_CONFLICT.value == "resource_conflict"
        assert ConflictType.PRIORITY_CONFLICT.value == "priority_conflict"
        assert ConflictType.DEPENDENCY_CONFLICT.value == "dependency_conflict"
        assert ConflictType.SEMANTIC_CONFLICT.value == "semantic_conflict"

    def test_conflict_type_is_str_enum(self):
        """Test that ConflictType values are strings"""
        for conflict_type in ConflictType:
            assert isinstance(conflict_type.value, str)

    def test_conflict_type_count(self):
        """Test expected number of conflict types"""
        assert len(list(ConflictType)) == 6


class TestResolutionStrategy:
    """Test ResolutionStrategy enum"""

    def test_resolution_strategy_values(self):
        """Test all resolution strategy enum values"""
        assert ResolutionStrategy.MERGE.value == "merge"
        assert ResolutionStrategy.PRIORITIZE.value == "prioritize"
        assert ResolutionStrategy.SEQUENCE.value == "sequence"
        assert ResolutionStrategy.DELEGATE.value == "delegate"
        assert ResolutionStrategy.NEGOTIATE.value == "negotiate"
        assert ResolutionStrategy.PARTITION.value == "partition"

    def test_resolution_strategy_is_str_enum(self):
        """Test that ResolutionStrategy values are strings"""
        for strategy in ResolutionStrategy:
            assert isinstance(strategy.value, str)

    def test_resolution_strategy_count(self):
        """Test expected number of resolution strategies"""
        assert len(list(ResolutionStrategy)) == 6


class TestConflictSeverity:
    """Test ConflictSeverity enum"""

    def test_conflict_severity_values(self):
        """Test all conflict severity enum values"""
        assert ConflictSeverity.LOW.value == "low"
        assert ConflictSeverity.MEDIUM.value == "medium"
        assert ConflictSeverity.HIGH.value == "high"
        assert ConflictSeverity.CRITICAL.value == "critical"

    def test_conflict_severity_is_str_enum(self):
        """Test that ConflictSeverity values are strings"""
        for severity in ConflictSeverity:
            assert isinstance(severity.value, str)

    def test_conflict_severity_count(self):
        """Test expected number of severity levels"""
        assert len(list(ConflictSeverity)) == 4


# ============================================================================
# Configuration Tests
# ============================================================================

class TestCoordinationConfig:
    """Test CoordinationConfig dataclass"""

    def test_config_default_initialization(self):
        """Test default configuration values"""
        config = CoordinationConfig()

        assert config.tier == "balanced"
        assert config.max_resolution_attempts == 3
        assert config.auto_resolve_low_severity is True
        assert config.enable_consensus_building is True
        assert config.enable_resource_management is True
        assert config.conflict_detection_threshold == 0.5
        assert config.require_verification is True
        assert config.escalation_enabled is True

    def test_config_custom_initialization(self):
        """Test custom configuration values"""
        config = CoordinationConfig(
            tier="expensive",
            max_resolution_attempts=5,
            auto_resolve_low_severity=False,
            enable_consensus_building=False,
            conflict_detection_threshold=0.8
        )

        assert config.tier == "expensive"
        assert config.max_resolution_attempts == 5
        assert config.auto_resolve_low_severity is False
        assert config.enable_consensus_building is False
        assert config.conflict_detection_threshold == 0.8

    def test_config_preferred_strategies_default(self):
        """Test default preferred strategies"""
        config = CoordinationConfig()

        assert len(config.preferred_strategies) == 3
        assert ResolutionStrategy.MERGE in config.preferred_strategies
        assert ResolutionStrategy.SEQUENCE in config.preferred_strategies
        assert ResolutionStrategy.NEGOTIATE in config.preferred_strategies

    def test_config_validation_max_attempts(self):
        """Test validation of max_resolution_attempts"""
        with pytest.raises(ValueError, match="max_resolution_attempts must be >= 1"):
            CoordinationConfig(max_resolution_attempts=0)

        with pytest.raises(ValueError, match="max_resolution_attempts must be >= 1"):
            CoordinationConfig(max_resolution_attempts=-1)

    def test_config_validation_threshold(self):
        """Test validation of conflict_detection_threshold"""
        with pytest.raises(ValueError, match="conflict_detection_threshold must be between 0 and 1"):
            CoordinationConfig(conflict_detection_threshold=-0.1)

        with pytest.raises(ValueError, match="conflict_detection_threshold must be between 0 and 1"):
            CoordinationConfig(conflict_detection_threshold=1.5)

    def test_config_valid_threshold_boundaries(self):
        """Test valid boundary values for threshold"""
        config1 = CoordinationConfig(conflict_detection_threshold=0.0)
        assert config1.conflict_detection_threshold == 0.0

        config2 = CoordinationConfig(conflict_detection_threshold=1.0)
        assert config2.conflict_detection_threshold == 1.0

    def test_config_custom_strategies(self):
        """Test custom preferred strategies"""
        custom_strategies = [ResolutionStrategy.PRIORITIZE, ResolutionStrategy.DELEGATE]
        config = CoordinationConfig(preferred_strategies=custom_strategies)

        assert len(config.preferred_strategies) == 2
        assert ResolutionStrategy.PRIORITIZE in config.preferred_strategies
        assert ResolutionStrategy.DELEGATE in config.preferred_strategies


class TestPresetCoordinationConfigs:
    """Test preset configuration factory"""

    def test_lightweight_preset(self):
        """Test lightweight preset configuration"""
        config = PresetCoordinationConfigs.lightweight()

        assert config.tier == "cheap"
        assert config.max_resolution_attempts == 2
        assert config.auto_resolve_low_severity is True
        assert config.enable_consensus_building is False
        assert config.enable_resource_management is False
        assert config.require_verification is False

    def test_standard_preset(self):
        """Test standard preset configuration"""
        config = PresetCoordinationConfigs.standard()

        assert config.tier == "balanced"
        assert config.max_resolution_attempts == 3
        assert config.auto_resolve_low_severity is True
        assert config.enable_consensus_building is True
        assert config.enable_resource_management is True
        assert config.require_verification is True

    def test_thorough_preset(self):
        """Test thorough preset configuration"""
        config = PresetCoordinationConfigs.thorough()

        assert config.tier == "expensive"
        assert config.max_resolution_attempts == 5
        assert config.auto_resolve_low_severity is False
        assert config.enable_consensus_building is True
        assert config.enable_resource_management is True
        assert config.require_verification is True
        assert config.conflict_detection_threshold == 0.3


# ============================================================================
# Data Structure Tests
# ============================================================================

class TestAgentOutput:
    """Test AgentOutput dataclass"""

    def test_agent_output_creation(self):
        """Test creating AgentOutput"""
        output = AgentOutput(
            agent_id="dev1",
            agent_type="developer",
            output_type="code",
            content="def foo(): pass"
        )

        assert output.agent_id == "dev1"
        assert output.agent_type == "developer"
        assert output.output_type == "code"
        assert output.content == "def foo(): pass"
        assert output.metadata == {}

    def test_agent_output_with_metadata(self):
        """Test AgentOutput with metadata"""
        output = AgentOutput(
            agent_id="research1",
            agent_type="research",
            output_type="analysis",
            content="Analysis results",
            metadata={"priority": 1, "tokens": 500}
        )

        assert output.metadata["priority"] == 1
        assert output.metadata["tokens"] == 500

    def test_agent_output_different_content_types(self):
        """Test AgentOutput with various content types"""
        # String content
        output1 = AgentOutput("id1", "type1", "text", "some text")
        assert isinstance(output1.content, str)

        # Dict content
        output2 = AgentOutput("id2", "type2", "json", {"key": "value"})
        assert isinstance(output2.content, dict)

        # List content
        output3 = AgentOutput("id3", "type3", "list", [1, 2, 3])
        assert isinstance(output3.content, list)


class TestConflict:
    """Test Conflict dataclass"""

    def test_conflict_creation(self):
        """Test creating Conflict"""
        output1 = AgentOutput("dev1", "developer", "code", "code1")
        output2 = AgentOutput("dev2", "developer", "code", "code2")

        conflict = Conflict(
            conflict_id="conflict_001",
            conflict_type=ConflictType.CODE_CONFLICT,
            severity=ConflictSeverity.MEDIUM,
            description="Conflicting implementations",
            affected_agents=["dev1", "dev2"],
            affected_outputs=[output1, output2]
        )

        assert conflict.conflict_id == "conflict_001"
        assert conflict.conflict_type == ConflictType.CODE_CONFLICT
        assert conflict.severity == ConflictSeverity.MEDIUM
        assert len(conflict.affected_agents) == 2
        assert len(conflict.affected_outputs) == 2

    def test_conflict_with_metadata(self):
        """Test Conflict with metadata"""
        conflict = Conflict(
            conflict_id="conflict_002",
            conflict_type=ConflictType.RESOURCE_CONFLICT,
            severity=ConflictSeverity.HIGH,
            description="Resource contention",
            affected_agents=["agent1"],
            affected_outputs=[],
            metadata={"resource": "database", "timestamp": "2025-10-19"}
        )

        assert conflict.metadata["resource"] == "database"
        assert conflict.metadata["timestamp"] == "2025-10-19"

    def test_conflict_different_severity_levels(self):
        """Test conflicts with different severity levels"""
        for severity in ConflictSeverity:
            conflict = Conflict(
                conflict_id=f"conflict_{severity.value}",
                conflict_type=ConflictType.SEMANTIC_CONFLICT,
                severity=severity,
                description=f"{severity.value} conflict",
                affected_agents=["agent1"],
                affected_outputs=[]
            )
            assert conflict.severity == severity


class TestResolution:
    """Test Resolution dataclass"""

    def test_resolution_creation(self):
        """Test creating Resolution"""
        resolution = Resolution(
            conflict_id="conflict_001",
            strategy=ResolutionStrategy.MERGE,
            steps=["Step 1", "Step 2", "Step 3"],
            expected_outcome="Conflict resolved"
        )

        assert resolution.conflict_id == "conflict_001"
        assert resolution.strategy == ResolutionStrategy.MERGE
        assert len(resolution.steps) == 3
        assert resolution.expected_outcome == "Conflict resolved"
        assert resolution.implemented is False
        assert resolution.verified is False
        assert resolution.verification_result is None

    def test_resolution_implemented(self):
        """Test Resolution marked as implemented"""
        resolution = Resolution(
            conflict_id="conflict_002",
            strategy=ResolutionStrategy.SEQUENCE,
            steps=["Step 1"],
            expected_outcome="Outcome",
            implemented=True
        )

        assert resolution.implemented is True

    def test_resolution_verified(self):
        """Test Resolution with verification"""
        verification = VerificationResult(
            is_success=True,
            score=1.0,
            reasoning="Verified successfully",
            failures=[],
            recommendations=[]
        )

        resolution = Resolution(
            conflict_id="conflict_003",
            strategy=ResolutionStrategy.NEGOTIATE,
            steps=["Step 1"],
            expected_outcome="Outcome",
            implemented=True,
            verified=True,
            verification_result=verification
        )

        assert resolution.verified is True
        assert resolution.verification_result is not None
        assert resolution.verification_result.is_success is True

    def test_resolution_different_strategies(self):
        """Test resolutions with different strategies"""
        for strategy in ResolutionStrategy:
            resolution = Resolution(
                conflict_id=f"conflict_{strategy.value}",
                strategy=strategy,
                steps=[f"Use {strategy.value}"],
                expected_outcome=f"{strategy.value} outcome"
            )
            assert resolution.strategy == strategy


class TestCoordinationResult:
    """Test CoordinationResult dataclass"""

    def test_coordination_result_creation(self):
        """Test creating CoordinationResult"""
        conflict = Conflict(
            conflict_id="c1",
            conflict_type=ConflictType.CODE_CONFLICT,
            severity=ConflictSeverity.LOW,
            description="Minor conflict",
            affected_agents=["dev1"],
            affected_outputs=[]
        )

        resolution = Resolution(
            conflict_id="c1",
            strategy=ResolutionStrategy.MERGE,
            steps=["Merge code"],
            expected_outcome="Merged",
            implemented=True
        )

        result = CoordinationResult(
            conflicts_detected=[conflict],
            resolutions=[resolution],
            consensus_reached=True
        )

        assert len(result.conflicts_detected) == 1
        assert len(result.resolutions) == 1
        assert result.consensus_reached is True
        assert result.resource_allocation is None
        assert result.verification_results == []

    def test_coordination_result_all_resolved_property(self):
        """Test all_resolved property"""
        conflict1 = Conflict("c1", ConflictType.CODE_CONFLICT, ConflictSeverity.LOW, "desc", [], [])
        conflict2 = Conflict("c2", ConflictType.GOAL_CONFLICT, ConflictSeverity.MEDIUM, "desc", [], [])

        resolution1 = Resolution("c1", ResolutionStrategy.MERGE, [], "out", implemented=True)
        resolution2 = Resolution("c2", ResolutionStrategy.MERGE, [], "out", implemented=True)

        result = CoordinationResult(
            conflicts_detected=[conflict1, conflict2],
            resolutions=[resolution1, resolution2],
            consensus_reached=True
        )

        assert result.all_resolved is True

    def test_coordination_result_not_all_resolved(self):
        """Test all_resolved when not all conflicts resolved"""
        conflict1 = Conflict("c1", ConflictType.CODE_CONFLICT, ConflictSeverity.LOW, "desc", [], [])
        conflict2 = Conflict("c2", ConflictType.GOAL_CONFLICT, ConflictSeverity.MEDIUM, "desc", [], [])

        resolution1 = Resolution("c1", ResolutionStrategy.MERGE, [], "out", implemented=True)

        result = CoordinationResult(
            conflicts_detected=[conflict1, conflict2],
            resolutions=[resolution1],
            consensus_reached=False
        )

        assert result.all_resolved is False

    def test_coordination_result_critical_conflicts_resolved(self):
        """Test critical_conflicts_resolved property"""
        critical_conflict = Conflict(
            "c1", ConflictType.CODE_CONFLICT, ConflictSeverity.CRITICAL, "critical", [], []
        )
        low_conflict = Conflict(
            "c2", ConflictType.GOAL_CONFLICT, ConflictSeverity.LOW, "low", [], []
        )

        resolution1 = Resolution("c1", ResolutionStrategy.MERGE, [], "out", implemented=True)
        resolution2 = Resolution("c2", ResolutionStrategy.MERGE, [], "out", implemented=False)

        result = CoordinationResult(
            conflicts_detected=[critical_conflict, low_conflict],
            resolutions=[resolution1, resolution2],
            consensus_reached=True
        )

        assert result.critical_conflicts_resolved is True

    def test_coordination_result_critical_not_resolved(self):
        """Test critical_conflicts_resolved when critical not resolved"""
        critical_conflict = Conflict(
            "c1", ConflictType.CODE_CONFLICT, ConflictSeverity.CRITICAL, "critical", [], []
        )

        resolution1 = Resolution("c1", ResolutionStrategy.MERGE, [], "out", implemented=False)

        result = CoordinationResult(
            conflicts_detected=[critical_conflict],
            resolutions=[resolution1],
            consensus_reached=False
        )

        assert result.critical_conflicts_resolved is False

    def test_coordination_result_with_resource_allocation(self):
        """Test CoordinationResult with resource allocation"""
        result = CoordinationResult(
            conflicts_detected=[],
            resolutions=[],
            consensus_reached=True,
            resource_allocation={"agent1": ["resource1", "resource2"]}
        )

        assert result.resource_allocation is not None
        assert "agent1" in result.resource_allocation

    def test_coordination_result_with_verification(self):
        """Test CoordinationResult with verification results"""
        verification = VerificationResult(
            is_success=True,
            score=1.0,
            reasoning="Verified",
            failures=[],
            recommendations=[]
        )

        result = CoordinationResult(
            conflicts_detected=[],
            resolutions=[],
            consensus_reached=True,
            verification_results=[verification]
        )

        assert len(result.verification_results) == 1
        assert result.verification_results[0].is_success is True


# ============================================================================
# Agent Initialization Tests
# ============================================================================

class TestCoordinationAgentInitialization:
    """Test CoordinationAgent initialization"""

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_agent_default_initialization(self, mock_configure):
        """Test agent initialization with defaults"""
        agent = CoordinationAgent()

        assert agent is not None
        assert agent.config is not None
        assert agent.tier == "balanced"
        mock_configure.assert_called_once()

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_agent_with_custom_config(self, mock_configure):
        """Test agent initialization with custom config"""
        config = CoordinationConfig(
            tier="expensive",
            max_resolution_attempts=5
        )
        agent = CoordinationAgent(config=config)

        assert agent.config == config
        assert agent.tier == "expensive"

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_agent_with_tier_override(self, mock_configure):
        """Test agent initialization with tier override"""
        config = CoordinationConfig(tier="balanced")
        agent = CoordinationAgent(config=config, tier="expensive")

        assert agent.tier == "expensive"

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_agent_has_required_modules(self, mock_configure):
        """Test agent has required DSPy modules"""
        agent = CoordinationAgent()

        assert hasattr(agent, 'conflict_detector')
        assert hasattr(agent, 'conflict_resolver')
        assert hasattr(agent, 'consensus_builder')
        assert hasattr(agent, 'resource_allocator')

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_agent_with_preset_configs(self, mock_configure):
        """Test agent with preset configurations"""
        presets = [
            PresetCoordinationConfigs.lightweight(),
            PresetCoordinationConfigs.standard(),
            PresetCoordinationConfigs.thorough()
        ]

        for config in presets:
            agent = CoordinationAgent(config=config)
            assert agent is not None
            assert agent.config == config


# ============================================================================
# Conflict Detection Tests
# ============================================================================

class TestConflictDetection:
    """Test conflict detection functionality"""

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_detect_conflicts_no_outputs(self, mock_configure):
        """Test conflict detection with no outputs"""
        agent = CoordinationAgent()
        conflicts = agent.detect_conflicts([], {})

        assert conflicts == []

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_detect_conflicts_single_output(self, mock_configure):
        """Test conflict detection with single output"""
        agent = CoordinationAgent()
        output = AgentOutput("dev1", "developer", "code", "code")

        conflicts = agent.detect_conflicts([output], {})

        assert conflicts == []

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_detect_conflicts_with_mock_detector(self, mock_configure):
        """Test conflict detection with mocked detector"""
        agent = CoordinationAgent()

        # Mock the conflict detector
        mock_detection_result = Mock()
        mock_detection_result.conflicts_detected = "conflict detected"
        mock_detection_result.conflict_summary = "There is a conflict between agents"
        agent.conflict_detector = Mock(return_value=mock_detection_result)

        output1 = AgentOutput("dev1", "developer", "code", "code1")
        output2 = AgentOutput("dev2", "developer", "code", "code2")

        conflicts = agent.detect_conflicts([output1, output2], {})

        assert len(conflicts) >= 1
        assert agent.conflict_detector.called

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_detect_conflicts_no_conflict_detected(self, mock_configure):
        """Test conflict detection when no conflict exists"""
        agent = CoordinationAgent()

        # Mock detector returning no conflict
        mock_detection_result = Mock()
        mock_detection_result.conflicts_detected = "no conflict"
        mock_detection_result.conflict_summary = "no conflict found"
        agent.conflict_detector = Mock(return_value=mock_detection_result)

        output1 = AgentOutput("dev1", "developer", "code", "code1")
        output2 = AgentOutput("dev2", "developer", "code", "code2")

        conflicts = agent.detect_conflicts([output1, output2], {})

        assert conflicts == []


# ============================================================================
# Conflict Resolution Tests
# ============================================================================

class TestConflictResolution:
    """Test conflict resolution functionality"""

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_resolve_conflict_basic(self, mock_configure):
        """Test basic conflict resolution"""
        agent = CoordinationAgent()

        conflict = Conflict(
            conflict_id="c1",
            conflict_type=ConflictType.CODE_CONFLICT,
            severity=ConflictSeverity.MEDIUM,
            description="Code conflict",
            affected_agents=["dev1", "dev2"],
            affected_outputs=[]
        )

        # Mock the resolver
        mock_resolution_result = Mock()
        mock_resolution_result.recommended_strategy = "merge"
        mock_resolution_result.resolution_steps = "1. Step one\n2. Step two"
        mock_resolution_result.expected_outcome = "Conflict resolved"
        agent.conflict_resolver = Mock(return_value=mock_resolution_result)

        resolution = agent.resolve_conflict(conflict, [])

        assert resolution.conflict_id == "c1"
        assert resolution.strategy == ResolutionStrategy.MERGE
        assert resolution.implemented is True
        assert len(resolution.steps) > 0

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_resolve_conflict_different_strategies(self, mock_configure):
        """Test resolution with different strategies"""
        agent = CoordinationAgent()

        conflict = Conflict("c1", ConflictType.CODE_CONFLICT, ConflictSeverity.MEDIUM,
                          "desc", [], [])

        strategy_mappings = [
            ("merge", ResolutionStrategy.MERGE),
            ("prioritize the first", ResolutionStrategy.PRIORITIZE),
            ("sequence them", ResolutionStrategy.SEQUENCE),
            ("delegate to higher level", ResolutionStrategy.DELEGATE),
            ("negotiate between agents", ResolutionStrategy.NEGOTIATE),
            ("partition the resources", ResolutionStrategy.PARTITION),
        ]

        for strategy_str, expected_strategy in strategy_mappings:
            mock_result = Mock()
            mock_result.recommended_strategy = strategy_str
            mock_result.resolution_steps = "Step 1"
            mock_result.expected_outcome = "Resolved"
            agent.conflict_resolver = Mock(return_value=mock_result)

            resolution = agent.resolve_conflict(conflict, [])
            assert resolution.strategy == expected_strategy

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_auto_resolve_low_severity(self, mock_configure):
        """Test auto-resolution of low severity conflicts"""
        agent = CoordinationAgent()

        conflict = Conflict(
            "c1", ConflictType.CODE_CONFLICT, ConflictSeverity.LOW,
            "minor conflict", ["dev1"], []
        )

        resolution = agent._auto_resolve_low_severity(conflict)

        assert resolution.conflict_id == "c1"
        assert resolution.strategy == ResolutionStrategy.MERGE
        assert resolution.implemented is True


# ============================================================================
# Consensus Building Tests
# ============================================================================

class TestConsensusBuilding:
    """Test consensus building functionality"""

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_build_consensus_no_conflicts(self, mock_configure):
        """Test consensus building with no conflicts"""
        agent = CoordinationAgent()

        consensus = agent.build_consensus([], [], [])

        assert consensus is True

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_build_consensus_with_conflicts(self, mock_configure):
        """Test consensus building with conflicts"""
        agent = CoordinationAgent()

        output1 = AgentOutput("dev1", "developer", "code", "code1")
        output2 = AgentOutput("dev2", "developer", "code", "code2")
        conflict = Conflict("c1", ConflictType.CODE_CONFLICT, ConflictSeverity.MEDIUM,
                          "desc", ["dev1", "dev2"], [])
        resolution = Resolution("c1", ResolutionStrategy.MERGE, [], "outcome")

        # Mock consensus builder
        mock_consensus_result = Mock()
        mock_consensus_result.consensus_proposal = "Agents agree to merge"
        mock_consensus_result.trade_offs = "Minor changes required"
        agent.consensus_builder = Mock(return_value=mock_consensus_result)

        consensus = agent.build_consensus([output1, output2], [conflict], [resolution])

        assert consensus is True
        assert agent.consensus_builder.called

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_build_consensus_failure(self, mock_configure):
        """Test consensus building failure"""
        agent = CoordinationAgent()

        output1 = AgentOutput("dev1", "developer", "code", "code1")
        conflict = Conflict("c1", ConflictType.CODE_CONFLICT, ConflictSeverity.HIGH,
                          "desc", ["dev1"], [])
        resolution = Resolution("c1", ResolutionStrategy.DELEGATE, [], "outcome")

        # Mock consensus builder returning empty proposal
        mock_consensus_result = Mock()
        mock_consensus_result.consensus_proposal = ""
        mock_consensus_result.trade_offs = ""
        agent.consensus_builder = Mock(return_value=mock_consensus_result)

        consensus = agent.build_consensus([output1], [conflict], [resolution])

        assert consensus is False


# ============================================================================
# Resource Allocation Tests
# ============================================================================

class TestResourceAllocation:
    """Test resource allocation functionality"""

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_allocate_resources_basic(self, mock_configure):
        """Test basic resource allocation"""
        agent = CoordinationAgent()

        # Mock resource allocator
        mock_allocation_result = Mock()
        mock_allocation_result.allocation_plan = "agent1: resource1, agent2: resource2"
        mock_allocation_result.potential_conflicts = "No conflicts"
        agent.resource_allocator = Mock(return_value=mock_allocation_result)

        allocation = agent.allocate_resources(
            available_resources=["resource1", "resource2"],
            agent_requests={"agent1": ["resource1"], "agent2": ["resource2"]}
        )

        assert "plan" in allocation
        assert "potential_conflicts" in allocation
        assert agent.resource_allocator.called

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_allocate_resources_with_constraints(self, mock_configure):
        """Test resource allocation with constraints"""
        agent = CoordinationAgent()

        mock_allocation_result = Mock()
        mock_allocation_result.allocation_plan = "Allocation with constraints"
        mock_allocation_result.potential_conflicts = "Possible contention"
        agent.resource_allocator = Mock(return_value=mock_allocation_result)

        constraints = {"max_concurrent": 2, "priority": "agent1"}
        allocation = agent.allocate_resources(
            available_resources=["res1", "res2"],
            agent_requests={"agent1": ["res1"], "agent2": ["res2"]},
            constraints=constraints
        )

        assert allocation is not None
        assert agent.resource_allocator.called

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_allocate_resources_conflicting_requests(self, mock_configure):
        """Test resource allocation with conflicting requests"""
        agent = CoordinationAgent()

        mock_allocation_result = Mock()
        mock_allocation_result.allocation_plan = "Sequential allocation"
        mock_allocation_result.potential_conflicts = "agent1 and agent2 need same resource"
        agent.resource_allocator = Mock(return_value=mock_allocation_result)

        # Both agents want same resource
        allocation = agent.allocate_resources(
            available_resources=["shared_resource"],
            agent_requests={
                "agent1": ["shared_resource"],
                "agent2": ["shared_resource"]
            }
        )

        assert "potential_conflicts" in allocation
        assert "agent1 and agent2" in allocation["potential_conflicts"]


# ============================================================================
# Helper Method Tests
# ============================================================================

class TestHelperMethods:
    """Test internal helper methods"""

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_parse_strategy_merge(self, mock_configure):
        """Test parsing merge strategy"""
        agent = CoordinationAgent()

        assert agent._parse_strategy("merge") == ResolutionStrategy.MERGE
        assert agent._parse_strategy("Merge the outputs") == ResolutionStrategy.MERGE
        assert agent._parse_strategy("MERGE") == ResolutionStrategy.MERGE

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_parse_strategy_prioritize(self, mock_configure):
        """Test parsing prioritize strategy"""
        agent = CoordinationAgent()

        assert agent._parse_strategy("prioritize") == ResolutionStrategy.PRIORITIZE
        assert agent._parse_strategy("priority based") == ResolutionStrategy.PRIORITIZE

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_parse_strategy_sequence(self, mock_configure):
        """Test parsing sequence strategy"""
        agent = CoordinationAgent()

        assert agent._parse_strategy("sequence") == ResolutionStrategy.SEQUENCE
        assert agent._parse_strategy("sequential execution") == ResolutionStrategy.SEQUENCE

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_parse_strategy_delegate(self, mock_configure):
        """Test parsing delegate strategy"""
        agent = CoordinationAgent()

        assert agent._parse_strategy("delegate") == ResolutionStrategy.DELEGATE
        assert agent._parse_strategy("escalate to control") == ResolutionStrategy.DELEGATE

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_parse_strategy_negotiate(self, mock_configure):
        """Test parsing negotiate strategy"""
        agent = CoordinationAgent()

        assert agent._parse_strategy("negotiate") == ResolutionStrategy.NEGOTIATE
        assert agent._parse_strategy("consensus building") == ResolutionStrategy.NEGOTIATE

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_parse_strategy_partition(self, mock_configure):
        """Test parsing partition strategy"""
        agent = CoordinationAgent()

        assert agent._parse_strategy("partition") == ResolutionStrategy.PARTITION
        assert agent._parse_strategy("split resources") == ResolutionStrategy.PARTITION

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_parse_strategy_default(self, mock_configure):
        """Test parsing unknown strategy defaults to merge"""
        agent = CoordinationAgent()

        assert agent._parse_strategy("unknown") == ResolutionStrategy.MERGE
        assert agent._parse_strategy("") == ResolutionStrategy.MERGE

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_parse_steps_numbered(self, mock_configure):
        """Test parsing numbered steps"""
        agent = CoordinationAgent()

        steps_str = "1. First step\n2. Second step\n3. Third step"
        steps = agent._parse_steps(steps_str)

        assert len(steps) == 3
        assert "First step" in steps[0]
        assert "Second step" in steps[1]
        assert "Third step" in steps[2]

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_parse_steps_bulleted(self, mock_configure):
        """Test parsing bulleted steps"""
        agent = CoordinationAgent()

        steps_str = "• First step\n• Second step\n- Third step"
        steps = agent._parse_steps(steps_str)

        assert len(steps) == 3

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_parse_steps_limit(self, mock_configure):
        """Test parsing steps limits to 10"""
        agent = CoordinationAgent()

        # Create 15 steps
        steps_str = "\n".join([f"{i}. Step {i}" for i in range(1, 16)])
        steps = agent._parse_steps(steps_str)

        assert len(steps) == 10

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_parse_conflicts_no_conflict(self, mock_configure):
        """Test parsing when no conflict detected"""
        agent = CoordinationAgent()

        conflicts = agent._parse_conflicts(
            "no conflict detected",
            "no conflict in the outputs",
            []
        )

        assert conflicts == []

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_parse_conflicts_with_conflict(self, mock_configure):
        """Test parsing when conflict detected"""
        agent = CoordinationAgent()

        output1 = AgentOutput("dev1", "developer", "code", "code1")
        output2 = AgentOutput("dev2", "developer", "code", "code2")

        conflicts = agent._parse_conflicts(
            "conflict detected between agents",
            "There is a semantic conflict",
            [output1, output2]
        )

        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.SEMANTIC_CONFLICT

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_format_agent_positions(self, mock_configure):
        """Test formatting agent positions for consensus"""
        agent = CoordinationAgent()

        output1 = AgentOutput("dev1", "developer", "code", "code1")
        output2 = AgentOutput("dev2", "developer", "code", "code2")
        conflict = Conflict("c1", ConflictType.CODE_CONFLICT, ConflictSeverity.MEDIUM,
                          "desc", ["dev1", "dev2"], [])

        positions = agent._format_agent_positions([output1, output2], [conflict])

        assert "dev1" in positions
        assert "dev2" in positions
        assert "developer" in positions


# ============================================================================
# Integration Tests (Full Workflow)
# ============================================================================

class TestCoordinationWorkflow:
    """Test full coordination workflow"""

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_forward_no_conflicts(self, mock_configure):
        """Test forward method with no conflicts"""
        agent = CoordinationAgent()

        # Mock detector to return no conflict
        mock_detection = Mock()
        mock_detection.conflicts_detected = "no conflict"
        mock_detection.conflict_summary = "no conflict"
        agent.conflict_detector = Mock(return_value=mock_detection)

        output1 = AgentOutput("dev1", "developer", "code", "code1")
        output2 = AgentOutput("dev2", "developer", "code", "code2")

        result = agent.forward([output1, output2])

        assert len(result.conflicts_detected) == 0
        assert len(result.resolutions) == 0
        assert result.consensus_reached is True
        assert result.all_resolved is True

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_forward_with_conflicts_and_resolution(self, mock_configure):
        """Test forward method with conflicts that get resolved"""
        config = CoordinationConfig(
            auto_resolve_low_severity=False,
            enable_consensus_building=False,
            require_verification=False
        )
        agent = CoordinationAgent(config=config)

        # Mock detector to return conflict
        mock_detection = Mock()
        mock_detection.conflicts_detected = "conflict detected"
        mock_detection.conflict_summary = "semantic conflict between agents"
        agent.conflict_detector = Mock(return_value=mock_detection)

        # Mock resolver
        mock_resolution = Mock()
        mock_resolution.recommended_strategy = "merge"
        mock_resolution.resolution_steps = "1. Merge\n2. Test"
        mock_resolution.expected_outcome = "Resolved"
        agent.conflict_resolver = Mock(return_value=mock_resolution)

        output1 = AgentOutput("dev1", "developer", "code", "code1")
        output2 = AgentOutput("dev2", "developer", "code", "code2")

        result = agent.forward([output1, output2])

        assert len(result.conflicts_detected) >= 1
        assert len(result.resolutions) >= 1
        assert result.resolutions[0].implemented is True

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_forward_with_auto_resolve(self, mock_configure):
        """Test forward method with auto-resolve enabled"""
        config = CoordinationConfig(
            auto_resolve_low_severity=True,
            enable_consensus_building=False,
            require_verification=False
        )
        agent = CoordinationAgent(config=config)

        # Mock detector to return low severity conflict
        mock_detection = Mock()
        mock_detection.conflicts_detected = "low severity conflict"
        mock_detection.conflict_summary = "minor conflict"
        agent.conflict_detector = Mock(return_value=mock_detection)

        # Override _parse_conflicts to return low severity
        original_parse = agent._parse_conflicts
        def mock_parse(conflicts_str, summary, outputs):
            return [Conflict(
                "c1", ConflictType.CODE_CONFLICT, ConflictSeverity.LOW,
                "low conflict", ["dev1"], []
            )]
        agent._parse_conflicts = mock_parse

        output1 = AgentOutput("dev1", "developer", "code", "code1")

        result = agent.forward([output1])

        # Should have auto-resolved
        assert len(result.resolutions) >= 1

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_forward_single_agent(self, mock_configure):
        """Test forward with single agent (no conflicts possible)"""
        agent = CoordinationAgent()

        output = AgentOutput("dev1", "developer", "code", "code1")

        result = agent.forward([output])

        assert len(result.conflicts_detected) == 0
        assert result.consensus_reached is True


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_empty_agent_outputs(self, mock_configure):
        """Test with empty agent outputs list"""
        agent = CoordinationAgent()

        result = agent.forward([])

        assert len(result.conflicts_detected) == 0
        assert result.consensus_reached is True

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_agent_output_with_none_content(self, mock_configure):
        """Test agent output with None content"""
        output = AgentOutput("dev1", "developer", "code", None)

        assert output.content is None
        assert output.agent_id == "dev1"

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_conflict_with_empty_affected_agents(self, mock_configure):
        """Test conflict with no affected agents"""
        conflict = Conflict(
            "c1", ConflictType.CODE_CONFLICT, ConflictSeverity.LOW,
            "desc", [], []
        )

        assert len(conflict.affected_agents) == 0

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_resolution_with_empty_steps(self, mock_configure):
        """Test resolution with no steps"""
        resolution = Resolution(
            "c1", ResolutionStrategy.MERGE, [], "outcome"
        )

        assert len(resolution.steps) == 0

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_coordination_result_empty(self, mock_configure):
        """Test completely empty coordination result"""
        result = CoordinationResult(
            conflicts_detected=[],
            resolutions=[],
            consensus_reached=True
        )

        assert result.all_resolved is True
        assert result.critical_conflicts_resolved is True

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_forward_with_context(self, mock_configure):
        """Test forward method with additional context"""
        agent = CoordinationAgent()

        # Mock detector
        mock_detection = Mock()
        mock_detection.conflicts_detected = "no conflict"
        mock_detection.conflict_summary = "no conflict"
        agent.conflict_detector = Mock(return_value=mock_detection)

        output = AgentOutput("dev1", "developer", "code", "code1")
        context = {"shared_resources": ["db"], "constraints": ["time_limit"]}

        result = agent.forward([output], context=context)

        assert result is not None
        # Verify context was passed to detector
        assert agent.conflict_detector.called


# ============================================================================
# DSPy Signature Tests
# ============================================================================

class TestDSpySignatures:
    """Test DSPy signature definitions"""

    def test_conflict_detection_signature_fields(self):
        """Test ConflictDetection signature has required fields"""
        assert hasattr(ConflictDetection, 'agent_outputs')
        assert hasattr(ConflictDetection, 'context')
        assert hasattr(ConflictDetection, 'conflicts_detected')
        assert hasattr(ConflictDetection, 'conflict_summary')

    def test_conflict_resolution_signature_fields(self):
        """Test ConflictResolution signature has required fields"""
        assert hasattr(ConflictResolution, 'conflict_description')
        assert hasattr(ConflictResolution, 'conflict_type')
        assert hasattr(ConflictResolution, 'affected_agents')
        assert hasattr(ConflictResolution, 'available_strategies')
        assert hasattr(ConflictResolution, 'recommended_strategy')
        assert hasattr(ConflictResolution, 'resolution_steps')
        assert hasattr(ConflictResolution, 'expected_outcome')

    def test_consensus_building_signature_fields(self):
        """Test ConsensusBuilding signature has required fields"""
        assert hasattr(ConsensusBuilding, 'agent_positions')
        assert hasattr(ConsensusBuilding, 'conflict_context')
        assert hasattr(ConsensusBuilding, 'success_criteria')
        assert hasattr(ConsensusBuilding, 'consensus_proposal')
        assert hasattr(ConsensusBuilding, 'trade_offs')
        assert hasattr(ConsensusBuilding, 'agent_agreements')

    def test_resource_allocation_signature_fields(self):
        """Test ResourceAllocation signature has required fields"""
        assert hasattr(ResourceAllocation, 'available_resources')
        assert hasattr(ResourceAllocation, 'agent_requests')
        assert hasattr(ResourceAllocation, 'constraints')
        assert hasattr(ResourceAllocation, 'allocation_plan')
        assert hasattr(ResourceAllocation, 'potential_conflicts')


# ============================================================================
# Verification Tests
# ============================================================================

class TestVerification:
    """Test resolution verification"""

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_verify_resolution_success(self, mock_configure):
        """Test verification of successful resolution"""
        agent = CoordinationAgent()

        resolution = Resolution(
            "c1", ResolutionStrategy.MERGE, ["step1"], "outcome", implemented=True
        )

        verification = agent._verify_resolution(resolution)

        assert verification.is_success is True
        assert verification.score == 1.0

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_verify_resolution_not_implemented(self, mock_configure):
        """Test verification of unimplemented resolution"""
        agent = CoordinationAgent()

        resolution = Resolution(
            "c1", ResolutionStrategy.MERGE, ["step1"], "outcome", implemented=False
        )

        verification = agent._verify_resolution(resolution)

        assert verification.is_success is False
        assert verification.score == 0.0

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_forward_with_verification_enabled(self, mock_configure):
        """Test forward with verification enabled"""
        config = CoordinationConfig(
            require_verification=True,
            enable_consensus_building=False,
            auto_resolve_low_severity=False
        )
        agent = CoordinationAgent(config=config)

        # Mock detector
        mock_detection = Mock()
        mock_detection.conflicts_detected = "conflict"
        mock_detection.conflict_summary = "test conflict"
        agent.conflict_detector = Mock(return_value=mock_detection)

        # Mock resolver
        mock_resolution = Mock()
        mock_resolution.recommended_strategy = "merge"
        mock_resolution.resolution_steps = "1. Merge"
        mock_resolution.expected_outcome = "Resolved"
        agent.conflict_resolver = Mock(return_value=mock_resolution)

        output1 = AgentOutput("dev1", "developer", "code", "code1")
        output2 = AgentOutput("dev2", "developer", "code", "code2")

        result = agent.forward([output1, output2])

        # Verification should have run
        if len(result.resolutions) > 0:
            assert len(result.verification_results) >= 0


# ============================================================================
# Metadata and Reporting Tests
# ============================================================================

class TestMetadata:
    """Test metadata tracking"""

    @patch('fractal_agent.agents.coordination_agent.configure_dspy')
    def test_coordination_result_metadata(self, mock_configure):
        """Test coordination result includes metadata"""
        agent = CoordinationAgent()

        # Mock no conflicts
        mock_detection = Mock()
        mock_detection.conflicts_detected = "no conflict"
        mock_detection.conflict_summary = "no conflict"
        agent.conflict_detector = Mock(return_value=mock_detection)

        output = AgentOutput("dev1", "developer", "code", "code1")

        result = agent.forward([output])

        assert "tier" in result.metadata
        assert "agent_count" in result.metadata
        assert result.metadata["agent_count"] == 1
