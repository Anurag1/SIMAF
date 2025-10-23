"""
Unit tests for ResearchConfig

Tests configuration validation and tier-based model selection.
"""

import pytest
from fractal_agent.agents.research_config import ResearchConfig, PresetConfigs


class TestResearchConfig:
    """Test ResearchConfig initialization and validation"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ResearchConfig()

        assert config.planning_tier == "expensive"
        assert config.research_tier == "cheap"
        assert config.synthesis_tier == "balanced"
        assert config.validation_tier == "balanced"
        assert config.max_tokens is None
        assert config.temperature == 0.7

    def test_custom_config(self):
        """Test custom configuration"""
        config = ResearchConfig(
            planning_tier="balanced",
            research_tier="cheap",
            synthesis_tier="expensive",
            validation_tier="cheap",
            max_tokens=1000,
            temperature=0.5
        )

        assert config.planning_tier == "balanced"
        assert config.research_tier == "cheap"
        assert config.synthesis_tier == "expensive"
        assert config.validation_tier == "cheap"
        assert config.max_tokens == 1000
        assert config.temperature == 0.5

    def test_config_repr(self):
        """Test config string representation"""
        config = ResearchConfig()
        repr_str = repr(config)

        assert "ResearchConfig" in repr_str
        assert "planning_tier='expensive'" in repr_str
        assert "research_tier='cheap'" in repr_str

    @pytest.mark.parametrize("tier", ["cheap", "balanced", "expensive", "premium"])
    def test_valid_tiers(self, tier):
        """Test all valid tier values"""
        config = ResearchConfig(
            planning_tier=tier,
            research_tier=tier,
            synthesis_tier=tier,
            validation_tier=tier
        )

        assert config.planning_tier == tier
        assert config.research_tier == tier
        assert config.synthesis_tier == tier
        assert config.validation_tier == tier

    def test_temperature_range(self):
        """Test temperature validation"""
        # Valid temperatures
        config = ResearchConfig(temperature=0.0)
        assert config.temperature == 0.0

        config = ResearchConfig(temperature=1.0)
        assert config.temperature == 1.0

        config = ResearchConfig(temperature=0.5)
        assert config.temperature == 0.5

    def test_max_tokens_validation(self):
        """Test max_tokens accepts None and positive integers"""
        config = ResearchConfig(max_tokens=None)
        assert config.max_tokens is None

        config = ResearchConfig(max_tokens=1000)
        assert config.max_tokens == 1000

        config = ResearchConfig(max_tokens=10000)
        assert config.max_tokens == 10000

    def test_config_str_with_max_tokens(self):
        """Test __str__ method with max_tokens set"""
        config = ResearchConfig(max_tokens=4000)
        str_repr = str(config)

        assert "ResearchConfig" in str_repr
        assert "planning=expensive" in str_repr
        assert "research=cheap" in str_repr
        assert "synthesis=balanced" in str_repr
        assert "validation=balanced" in str_repr
        assert "max_tokens=4000" in str_repr
        assert "temp=0.7" in str_repr

    def test_config_str_without_max_tokens(self):
        """Test __str__ method without max_tokens (unlimited)"""
        config = ResearchConfig()
        str_repr = str(config)

        assert "ResearchConfig" in str_repr
        assert "unlimited tokens" in str_repr
        assert "temp=0.7" in str_repr

    def test_config_str_custom_temperature(self):
        """Test __str__ with custom temperature"""
        config = ResearchConfig(temperature=0.9)
        str_repr = str(config)

        assert "temp=0.9" in str_repr

    def test_config_equality(self):
        """Test config instances with same values are equal"""
        config1 = ResearchConfig(planning_tier="premium", max_tokens=2000)
        config2 = ResearchConfig(planning_tier="premium", max_tokens=2000)

        assert config1 == config2

    def test_config_inequality(self):
        """Test config instances with different values are not equal"""
        config1 = ResearchConfig(planning_tier="premium")
        config2 = ResearchConfig(planning_tier="cheap")

        assert config1 != config2


class TestPresetConfigs:
    """Test PresetConfigs factory methods"""

    def test_default_preset(self):
        """Test default preset configuration"""
        config = PresetConfigs.default()

        assert isinstance(config, ResearchConfig)
        assert config.planning_tier == "expensive"
        assert config.research_tier == "cheap"
        assert config.synthesis_tier == "balanced"
        assert config.validation_tier == "balanced"
        assert config.max_tokens is None

    def test_cost_optimized_preset(self):
        """Test cost-optimized preset uses cheap/balanced tiers"""
        config = PresetConfigs.cost_optimized()

        assert isinstance(config, ResearchConfig)
        assert config.planning_tier == "balanced"
        assert config.research_tier == "cheap"
        assert config.synthesis_tier == "cheap"
        assert config.validation_tier == "cheap"

    def test_quality_optimized_preset(self):
        """Test quality-optimized preset uses expensive/premium tiers"""
        config = PresetConfigs.quality_optimized()

        assert isinstance(config, ResearchConfig)
        assert config.planning_tier == "premium"
        assert config.research_tier == "expensive"
        assert config.synthesis_tier == "premium"
        assert config.validation_tier == "expensive"

    def test_fast_iteration_preset(self):
        """Test fast iteration preset has token limits"""
        config = PresetConfigs.fast_iteration()

        assert isinstance(config, ResearchConfig)
        assert config.planning_tier == "cheap"
        assert config.research_tier == "cheap"
        assert config.synthesis_tier == "balanced"
        assert config.validation_tier == "cheap"
        assert config.max_tokens == 2000  # Should have token limit

    def test_all_presets_are_unique(self):
        """Test that all presets produce different configurations"""
        default = PresetConfigs.default()
        cost = PresetConfigs.cost_optimized()
        quality = PresetConfigs.quality_optimized()
        fast = PresetConfigs.fast_iteration()

        # All should be different
        assert default != cost
        assert default != quality
        assert default != fast
        assert cost != quality
        assert cost != fast
        assert quality != fast

    def test_presets_have_valid_tiers(self):
        """Test all presets use valid tiers"""
        valid_tiers = {"cheap", "balanced", "expensive", "premium"}
        presets = [
            PresetConfigs.default(),
            PresetConfigs.cost_optimized(),
            PresetConfigs.quality_optimized(),
            PresetConfigs.fast_iteration()
        ]

        for config in presets:
            assert config.planning_tier in valid_tiers
            assert config.research_tier in valid_tiers
            assert config.synthesis_tier in valid_tiers
            assert config.validation_tier in valid_tiers
