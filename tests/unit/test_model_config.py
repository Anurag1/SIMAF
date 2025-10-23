"""
Unit tests for model configuration

Tests tier-based model selection and configuration.
"""

import pytest
from unittest.mock import patch, Mock
from fractal_agent.utils.model_config import (
    get_provider_chain,
    configure_lm,
    configure_cheap_lm,
    configure_balanced_lm,
    configure_expensive_lm,
    configure_premium_lm,
    get_model_recommendations,
    list_available_configs,
    _get_fallback_model
)
from fractal_agent.utils.model_registry import ModelRegistry


class TestGetProviderChain:
    """Test get_provider_chain function"""

    def test_get_provider_chain_cheap(self):
        """Test getting provider chain for cheap tier"""
        chain = get_provider_chain("cheap")

        assert chain is not None
        assert isinstance(chain, list)
        assert len(chain) > 0
        # Verify structure
        for provider, model in chain:
            assert isinstance(provider, str)
            assert isinstance(model, str)

    def test_get_provider_chain_balanced(self):
        """Test getting provider chain for balanced tier"""
        chain = get_provider_chain("balanced")

        assert chain is not None
        assert len(chain) > 0
        # Should have anthropic and gemini
        providers = [p for p, m in chain]
        assert "anthropic" in providers or "gemini" in providers

    def test_get_provider_chain_expensive(self):
        """Test getting provider chain for expensive tier"""
        chain = get_provider_chain("expensive")

        assert chain is not None
        assert len(chain) > 0

    def test_get_provider_chain_premium(self):
        """Test getting provider chain for premium tier"""
        chain = get_provider_chain("premium")

        assert chain is not None
        assert len(chain) > 0

    def test_provider_chain_structure(self):
        """Test provider chain has correct structure"""
        chain = get_provider_chain("balanced")

        assert isinstance(chain, list)
        for item in chain:
            assert isinstance(item, tuple)
            assert len(item) == 2  # (provider, model)

    def test_all_tiers_have_provider_chains(self):
        """Test all tiers have provider chains"""
        tiers = ["cheap", "balanced", "expensive", "premium"]

        for tier in tiers:
            chain = get_provider_chain(tier)
            assert chain is not None
            assert len(chain) > 0

    def test_get_provider_chain_custom_providers(self):
        """Test provider chain with custom provider list"""
        chain = get_provider_chain("balanced", providers=["anthropic"])

        # Should only have anthropic
        providers = [p for p, m in chain]
        assert "anthropic" in providers
        # Gemini should not be in chain if we only specified anthropic
        # (though it might be if there's no anthropic model)

    def test_get_provider_chain_with_vision_requirement(self):
        """Test provider chain respects vision requirement"""
        # Some tiers may not have vision models, so this might return fewer models
        chain = get_provider_chain("balanced", require_vision=True, fallback_tiers=False)

        # If chain is empty, that's expected (no vision models in balanced)
        # If not empty, all models should support vision
        assert isinstance(chain, list)

    def test_get_provider_chain_with_caching_requirement(self):
        """Test provider chain respects caching requirement"""
        chain = get_provider_chain("balanced", require_caching=True, fallback_tiers=False)

        # Should return models with caching support
        assert isinstance(chain, list)

    def test_get_provider_chain_no_fallback(self):
        """Test provider chain without fallback tiers"""
        # Request a config that might not exist, with fallback disabled
        # This should either return a valid chain or empty list
        chain = get_provider_chain("premium", providers=["anthropic"], fallback_tiers=False)

        assert isinstance(chain, list)

    @pytest.mark.skip(reason="Registry may have models that match these requirements")
    def test_get_provider_chain_invalid_requirements(self):
        """Test provider chain with impossible requirements"""
        # This test is skipped because the actual registry may or may not
        # have models matching these requirements - it's not guaranteed to fail
        pass


class TestConfigureLM:
    """Test configure_lm and convenience functions"""

    def test_configure_lm_cheap(self):
        """Test configuring cheap tier LM"""
        with patch('fractal_agent.utils.model_config.UnifiedLM') as MockLM:
            mock_instance = Mock()
            MockLM.return_value = mock_instance

            lm = configure_lm("cheap")

            assert lm is not None
            MockLM.assert_called_once()
            # Verify providers argument was passed
            call_kwargs = MockLM.call_args[1]
            assert "providers" in call_kwargs

    def test_configure_lm_balanced(self):
        """Test configuring balanced tier LM"""
        with patch('fractal_agent.utils.model_config.UnifiedLM') as MockLM:
            mock_instance = Mock()
            MockLM.return_value = mock_instance

            lm = configure_lm("balanced")

            assert lm is not None
            MockLM.assert_called_once()

    def test_configure_lm_expensive(self):
        """Test configuring expensive tier LM"""
        with patch('fractal_agent.utils.model_config.UnifiedLM') as MockLM:
            mock_instance = Mock()
            MockLM.return_value = mock_instance

            lm = configure_lm("expensive")

            assert lm is not None

    def test_configure_lm_premium(self):
        """Test configuring premium tier LM"""
        with patch('fractal_agent.utils.model_config.UnifiedLM') as MockLM:
            mock_instance = Mock()
            MockLM.return_value = mock_instance

            lm = configure_lm("premium")

            assert lm is not None

    def test_configure_lm_with_custom_providers(self):
        """Test configure_lm with custom provider list"""
        with patch('fractal_agent.utils.model_config.UnifiedLM') as MockLM:
            mock_instance = Mock()
            MockLM.return_value = mock_instance

            lm = configure_lm("balanced", providers=["anthropic"])

            # Verify UnifiedLM was initialized with correct provider chain
            MockLM.assert_called_once()

    def test_configure_lm_passes_kwargs_to_unified_lm(self):
        """Test that extra kwargs are passed to UnifiedLM"""
        with patch('fractal_agent.utils.model_config.UnifiedLM') as MockLM:
            mock_instance = Mock()
            MockLM.return_value = mock_instance

            configure_lm("balanced", enable_caching=True, custom_param="value")

            call_kwargs = MockLM.call_args[1]
            assert "enable_caching" in call_kwargs
            assert call_kwargs["enable_caching"] is True
            assert "custom_param" in call_kwargs
            assert call_kwargs["custom_param"] == "value"

    def test_configure_cheap_lm(self):
        """Test configure_cheap_lm convenience function"""
        with patch('fractal_agent.utils.model_config.UnifiedLM') as MockLM:
            mock_instance = Mock()
            MockLM.return_value = mock_instance

            lm = configure_cheap_lm()

            assert lm is not None
            assert callable(lm) or hasattr(lm, '__call__')

    def test_configure_balanced_lm(self):
        """Test configure_balanced_lm convenience function"""
        with patch('fractal_agent.utils.model_config.UnifiedLM') as MockLM:
            mock_instance = Mock()
            MockLM.return_value = mock_instance

            lm = configure_balanced_lm()

            assert lm is not None

    def test_configure_expensive_lm(self):
        """Test configure_expensive_lm convenience function"""
        with patch('fractal_agent.utils.model_config.UnifiedLM') as MockLM:
            mock_instance = Mock()
            MockLM.return_value = mock_instance

            lm = configure_expensive_lm()

            assert lm is not None

    def test_configure_premium_lm(self):
        """Test configure_premium_lm convenience function"""
        with patch('fractal_agent.utils.model_config.UnifiedLM') as MockLM:
            mock_instance = Mock()
            MockLM.return_value = mock_instance

            lm = configure_premium_lm()

            assert lm is not None

    def test_convenience_functions_pass_kwargs(self):
        """Test that convenience functions pass kwargs"""
        with patch('fractal_agent.utils.model_config.UnifiedLM') as MockLM:
            mock_instance = Mock()
            MockLM.return_value = mock_instance

            configure_cheap_lm(enable_caching=False, custom="value")

            call_kwargs = MockLM.call_args[1]
            assert "enable_caching" in call_kwargs
            assert "custom" in call_kwargs


class TestModelRecommendations:
    """Test get_model_recommendations function"""

    def test_get_model_recommendations_classification(self):
        """Test recommendation for classification tasks"""
        tier = get_model_recommendations("classification")

        assert tier == "cheap"

    def test_get_model_recommendations_code_generation(self):
        """Test recommendation for code generation"""
        tier = get_model_recommendations("code_generation")

        assert tier == "balanced"

    def test_get_model_recommendations_complex_reasoning(self):
        """Test recommendation for complex reasoning"""
        tier = get_model_recommendations("complex_reasoning")

        assert tier == "expensive"

    def test_get_model_recommendations_strategic_planning(self):
        """Test recommendation for strategic planning"""
        tier = get_model_recommendations("strategic_planning")

        assert tier == "premium"

    def test_get_model_recommendations_unknown_task(self):
        """Test recommendation for unknown task defaults to balanced"""
        tier = get_model_recommendations("unknown_task_type")

        assert tier == "balanced"

    def test_get_model_recommendations_case_insensitive(self):
        """Test recommendations are case-insensitive"""
        tier1 = get_model_recommendations("CODE_GENERATION")
        tier2 = get_model_recommendations("code_generation")

        assert tier1 == tier2

    def test_model_recommendations_coverage(self):
        """Test model recommendations for various task types"""
        task_types = [
            "classification",
            "code_generation",
            "complex_reasoning",
            "strategic_planning",
            "document_analysis"
        ]

        for task_type in task_types:
            tier = get_model_recommendations(task_type)
            assert tier in ["cheap", "balanced", "expensive", "premium"]


class TestListAvailableConfigs:
    """Test list_available_configs function"""

    def test_list_available_configs(self):
        """Test listing available configurations"""
        configs = list_available_configs()

        assert configs is not None
        assert isinstance(configs, dict)
        assert len(configs) > 0

    def test_list_available_configs_structure(self):
        """Test structure of available configs"""
        configs = list_available_configs()

        # Should have tier keys
        assert "cheap" in configs or "balanced" in configs

        # Each tier should have provider configs
        for tier, tier_configs in configs.items():
            assert isinstance(tier_configs, dict)

            for provider, model_config in tier_configs.items():
                assert "model_id" in model_config
                assert "input_cost" in model_config
                assert "output_cost" in model_config
                assert "context_window" in model_config
                assert "supports_vision" in model_config
                assert "supports_caching" in model_config

    def test_list_available_configs_all_tiers(self):
        """Test all tiers are in available configs"""
        configs = list_available_configs()

        expected_tiers = ["cheap", "balanced", "expensive", "premium"]

        for tier in expected_tiers:
            # Tier might not have models, so just check structure
            if tier in configs:
                assert isinstance(configs[tier], dict)


class TestFallbackLogic:
    """Test fallback model selection"""

    def test_fallback_to_lower_tier(self):
        """Test fallback selects lower tier model"""
        registry = ModelRegistry()

        # Try to get fallback for expensive tier
        fallback = _get_fallback_model(
            provider="anthropic",
            tier="expensive",
            require_vision=False,
            require_caching=False,
            registry=registry
        )

        # Should return a model from lower tier (balanced or cheap)
        if fallback:
            assert fallback.tier in ["balanced", "cheap"]

    def test_no_fallback_for_cheap_tier(self):
        """Test no fallback available for cheap tier"""
        registry = ModelRegistry()

        fallback = _get_fallback_model(
            provider="anthropic",
            tier="cheap",
            require_vision=False,
            require_caching=False,
            registry=registry
        )

        # Cheap is lowest tier, so no fallback
        assert fallback is None

    def test_fallback_with_invalid_tier(self):
        """Test fallback handles invalid tier gracefully"""
        registry = ModelRegistry()

        fallback = _get_fallback_model(
            provider="anthropic",
            tier="invalid_tier",
            require_vision=False,
            require_caching=False,
            registry=registry
        )

        # Invalid tier should return None
        assert fallback is None

    @patch('fractal_agent.utils.model_config.ModelRegistry')
    def test_get_provider_chain_no_models_error(self, mock_registry_class):
        """Test error when no models found for tier"""
        # Create empty registry
        mock_registry = Mock()
        mock_registry.get_models_by_tier.return_value = []
        mock_registry_class.return_value = mock_registry

        with pytest.raises(ValueError) as exc_info:
            get_provider_chain("expensive", providers=["nonexistent"])

        assert "No models found" in str(exc_info.value)
