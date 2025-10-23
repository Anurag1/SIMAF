"""
Unit tests for DSPy integration

Tests FractalDSpyLM and DSPy configuration functions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import dspy
from fractal_agent.utils.dspy_integration import (
    FractalDSpyLM,
    configure_dspy,
    configure_dspy_cheap,
    configure_dspy_balanced,
    configure_dspy_expensive,
    configure_dspy_premium
)


@pytest.fixture
def mock_unified_lm():
    """Mock UnifiedLM instance"""
    mock_lm = Mock()
    mock_lm.return_value = {
        "text": "Mocked response from UnifiedLM",
        "tokens_used": 100,
        "cache_hit": False,
        "provider": "anthropic",
        "model": "claude-sonnet-4.5"
    }
    return mock_lm


@pytest.fixture
def mock_configure_lm():
    """Mock configure_lm function"""
    with patch('fractal_agent.utils.dspy_integration.configure_lm') as mock:
        yield mock


class TestFractalDSpyLMInitialization:
    """Test FractalDSpyLM initialization"""

    def test_fractal_dspy_lm_default_initialization(self, mock_configure_lm, mock_unified_lm):
        """Test FractalDSpyLM initializes with defaults"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM()

        assert lm.tier == "balanced"
        assert lm.providers is None
        assert lm.temperature == 0.7
        assert lm.max_tokens is None
        assert lm.unified_lm is not None
        assert len(lm.history) == 0

    def test_fractal_dspy_lm_custom_tier(self, mock_configure_lm, mock_unified_lm):
        """Test FractalDSpyLM with custom tier"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM(tier="expensive")

        assert lm.tier == "expensive"
        mock_configure_lm.assert_called_once()
        call_kwargs = mock_configure_lm.call_args[1]
        assert call_kwargs["tier"] == "expensive"

    def test_fractal_dspy_lm_custom_providers(self, mock_configure_lm, mock_unified_lm):
        """Test FractalDSpyLM with custom providers"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM(tier="balanced", providers=["anthropic"])

        assert lm.providers == ["anthropic"]
        call_kwargs = mock_configure_lm.call_args[1]
        assert call_kwargs["providers"] == ["anthropic"]

    def test_fractal_dspy_lm_custom_parameters(self, mock_configure_lm, mock_unified_lm):
        """Test FractalDSpyLM with custom parameters"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM(
            tier="balanced",
            max_tokens=500,
            temperature=0.5,
            require_caching=True
        )

        assert lm.max_tokens == 500
        assert lm.temperature == 0.5
        assert lm.require_caching is True

    def test_fractal_dspy_lm_inherits_from_dspy_lm(self, mock_configure_lm, mock_unified_lm):
        """Test FractalDSpyLM inherits from dspy.LM"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM()

        assert isinstance(lm, dspy.LM)


class TestFractalDSpyLMBasicRequest:
    """Test FractalDSpyLM basic_request method"""

    def test_basic_request_simple_prompt(self, mock_configure_lm, mock_unified_lm):
        """Test basic_request with simple prompt"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM()
        result = lm.basic_request("Test prompt")

        assert "choices" in result
        assert len(result["choices"]) > 0
        assert "text" in result["choices"][0]
        assert result["choices"][0]["text"] == "Mocked response from UnifiedLM"

    def test_basic_request_with_max_tokens(self, mock_configure_lm, mock_unified_lm):
        """Test basic_request respects max_tokens parameter"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM(max_tokens=200)
        lm.basic_request("Test prompt")

        # Verify unified_lm was called with max_tokens
        call_kwargs = mock_unified_lm.call_args[1]
        assert "max_tokens" in call_kwargs
        assert call_kwargs["max_tokens"] == 200

    def test_basic_request_with_temperature(self, mock_configure_lm, mock_unified_lm):
        """Test basic_request respects temperature parameter"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM(temperature=0.3)
        lm.basic_request("Test prompt")

        call_kwargs = mock_unified_lm.call_args[1]
        assert "temperature" in call_kwargs
        assert call_kwargs["temperature"] == 0.3

    def test_basic_request_overrides_temperature(self, mock_configure_lm, mock_unified_lm):
        """Test basic_request can override default temperature"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM(temperature=0.7)
        lm.basic_request("Test prompt", temperature=0.2)

        call_kwargs = mock_unified_lm.call_args[1]
        assert call_kwargs["temperature"] == 0.2

    def test_basic_request_tracks_history(self, mock_configure_lm, mock_unified_lm):
        """Test basic_request adds to history"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM()
        assert len(lm.history) == 0

        lm.basic_request("Test prompt 1")
        assert len(lm.history) == 1

        lm.basic_request("Test prompt 2")
        assert len(lm.history) == 2

        # Verify history content
        assert lm.history[0]["prompt"] == "Test prompt 1"
        assert lm.history[1]["prompt"] == "Test prompt 2"

    def test_basic_request_returns_dspy_format(self, mock_configure_lm, mock_unified_lm):
        """Test basic_request returns DSPy-compatible format"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM()
        result = lm.basic_request("Test")

        # Check DSPy format
        assert "choices" in result
        assert "usage" in result
        assert "metadata" in result
        assert "total_tokens" in result["usage"]
        assert "provider" in result["metadata"]
        assert "model" in result["metadata"]
        assert "tier" in result["metadata"]

    def test_basic_request_without_max_tokens(self, mock_configure_lm, mock_unified_lm):
        """Test basic_request when max_tokens is None"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM(max_tokens=None)
        lm.basic_request("Test")

        # max_tokens should not be in kwargs when None
        call_kwargs = mock_unified_lm.call_args[1]
        assert "max_tokens" not in call_kwargs


class TestFractalDSpyLMCall:
    """Test FractalDSpyLM __call__ method"""

    def test_call_with_prompt(self, mock_configure_lm, mock_unified_lm):
        """Test __call__ with simple prompt"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM()
        result = lm(prompt="Test prompt")

        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0] == "Mocked response from UnifiedLM"

    def test_call_with_messages(self, mock_configure_lm, mock_unified_lm):
        """Test __call__ with messages"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
        result = lm(messages=messages)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_call_converts_messages_to_prompt(self, mock_configure_lm, mock_unified_lm):
        """Test __call__ converts messages to prompt format"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        lm(messages=messages)

        # Verify basic_request was called with formatted prompt
        call_args = mock_unified_lm.call_args
        assert call_args is not None

    def test_call_requires_prompt_or_messages(self, mock_configure_lm, mock_unified_lm):
        """Test __call__ raises error without prompt or messages"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM()

        with pytest.raises(ValueError) as exc_info:
            lm()

        assert "Either prompt or messages must be provided" in str(exc_info.value)

    def test_call_prefers_prompt_over_messages(self, mock_configure_lm, mock_unified_lm):
        """Test __call__ uses prompt if both prompt and messages provided"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM()
        result = lm(prompt="Direct prompt", messages=[{"role": "user", "content": "Message"}])

        # Should use prompt directly, not messages
        assert isinstance(result, list)


class TestFractalDSpyLMMetrics:
    """Test FractalDSpyLM metrics tracking"""

    def test_get_metrics_empty_history(self, mock_configure_lm, mock_unified_lm):
        """Test get_metrics with no calls"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM()
        metrics = lm.get_metrics()

        assert metrics["total_calls"] == 0
        assert metrics["total_tokens"] == 0
        assert metrics["avg_tokens_per_call"] == 0
        assert isinstance(metrics["provider_distribution"], dict)

    def test_get_metrics_tracks_calls(self, mock_configure_lm, mock_unified_lm):
        """Test get_metrics tracks call count"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM()
        lm(prompt="Test 1")
        lm(prompt="Test 2")
        lm(prompt="Test 3")

        metrics = lm.get_metrics()

        assert metrics["total_calls"] == 3

    def test_get_metrics_tracks_tokens(self, mock_configure_lm, mock_unified_lm):
        """Test get_metrics tracks token usage"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM()
        lm(prompt="Test 1")
        lm(prompt="Test 2")

        metrics = lm.get_metrics()

        # Each call uses 100 tokens (from mock)
        assert metrics["total_tokens"] == 200
        assert metrics["avg_tokens_per_call"] == 100

    def test_get_metrics_provider_distribution(self, mock_configure_lm, mock_unified_lm):
        """Test get_metrics tracks provider distribution"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM()
        lm(prompt="Test 1")
        lm(prompt="Test 2")

        metrics = lm.get_metrics()

        assert "anthropic" in metrics["provider_distribution"]
        assert metrics["provider_distribution"]["anthropic"] == 2

    def test_get_metrics_includes_tier(self, mock_configure_lm, mock_unified_lm):
        """Test get_metrics includes tier information"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM(tier="expensive")
        lm(prompt="Test")  # Make at least one call to populate history
        metrics = lm.get_metrics()

        assert metrics["tier"] == "expensive"

    def test_clear_history(self, mock_configure_lm, mock_unified_lm):
        """Test clear_history clears the history"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM()
        lm(prompt="Test 1")
        lm(prompt="Test 2")
        assert len(lm.history) == 2

        lm.clear_history()
        assert len(lm.history) == 0


class TestFractalDSpyLMDeepcopy:
    """Test FractalDSpyLM deepcopy functionality for MIPRO"""

    def test_deepcopy_creates_new_instance(self, mock_configure_lm, mock_unified_lm):
        """Test deepcopy creates a new instance"""
        import copy
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM(tier="balanced", max_tokens=100)
        lm(prompt="Test")  # Add some history

        lm_copy = copy.deepcopy(lm)

        # Should be different objects
        assert lm is not lm_copy
        # But same configuration
        assert lm_copy.tier == "balanced"
        assert lm_copy.max_tokens == 100

    def test_deepcopy_does_not_copy_history(self, mock_configure_lm, mock_unified_lm):
        """Test deepcopy creates instance with empty history"""
        import copy
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM()
        lm(prompt="Test 1")
        lm(prompt="Test 2")
        assert len(lm.history) == 2

        lm_copy = copy.deepcopy(lm)

        # New instance should have empty history
        assert len(lm_copy.history) == 0

    def test_deepcopy_preserves_tier(self, mock_configure_lm, mock_unified_lm):
        """Test deepcopy preserves tier configuration"""
        import copy
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM(tier="expensive", temperature=0.3)
        lm_copy = copy.deepcopy(lm)

        assert lm_copy.tier == "expensive"
        assert lm_copy.temperature == 0.3


class TestConfigureDSPy:
    """Test configure_dspy and convenience functions"""

    def test_configure_dspy_default(self, mock_configure_lm, mock_unified_lm):
        """Test configure_dspy with defaults"""
        mock_configure_lm.return_value = mock_unified_lm

        with patch.object(dspy, 'configure') as mock_dspy_configure:
            lm = configure_dspy()

            assert isinstance(lm, FractalDSpyLM)
            assert lm.tier == "balanced"
            # Verify dspy.configure was called
            mock_dspy_configure.assert_called_once()

    def test_configure_dspy_custom_tier(self, mock_configure_lm, mock_unified_lm):
        """Test configure_dspy with custom tier"""
        mock_configure_lm.return_value = mock_unified_lm

        with patch.object(dspy, 'configure') as mock_dspy_configure:
            lm = configure_dspy(tier="expensive")

            assert lm.tier == "expensive"

    def test_configure_dspy_passes_kwargs(self, mock_configure_lm, mock_unified_lm):
        """Test configure_dspy passes additional kwargs"""
        mock_configure_lm.return_value = mock_unified_lm

        with patch.object(dspy, 'configure') as mock_dspy_configure:
            lm = configure_dspy(tier="balanced", max_tokens=500, temperature=0.2)

            assert lm.max_tokens == 500
            assert lm.temperature == 0.2

    def test_configure_dspy_cheap(self, mock_configure_lm, mock_unified_lm):
        """Test configure_dspy_cheap convenience function"""
        mock_configure_lm.return_value = mock_unified_lm

        with patch.object(dspy, 'configure'):
            lm = configure_dspy_cheap()

            assert lm.tier == "cheap"

    def test_configure_dspy_balanced(self, mock_configure_lm, mock_unified_lm):
        """Test configure_dspy_balanced convenience function"""
        mock_configure_lm.return_value = mock_unified_lm

        with patch.object(dspy, 'configure'):
            lm = configure_dspy_balanced()

            assert lm.tier == "balanced"

    def test_configure_dspy_expensive(self, mock_configure_lm, mock_unified_lm):
        """Test configure_dspy_expensive convenience function"""
        mock_configure_lm.return_value = mock_unified_lm

        with patch.object(dspy, 'configure'):
            lm = configure_dspy_expensive()

            assert lm.tier == "expensive"

    def test_configure_dspy_premium(self, mock_configure_lm, mock_unified_lm):
        """Test configure_dspy_premium convenience function"""
        mock_configure_lm.return_value = mock_unified_lm

        with patch.object(dspy, 'configure'):
            lm = configure_dspy_premium()

            assert lm.tier == "premium"

    def test_convenience_functions_pass_kwargs(self, mock_configure_lm, mock_unified_lm):
        """Test convenience functions pass kwargs"""
        mock_configure_lm.return_value = mock_unified_lm

        with patch.object(dspy, 'configure'):
            lm = configure_dspy_cheap(max_tokens=100, custom_param="value")

            assert lm.max_tokens == 100


class TestDSPyIntegration:
    """Test integration with DSPy modules"""

    def test_fractal_dspy_lm_compatible_with_dspy_predict(self, mock_configure_lm, mock_unified_lm):
        """Test FractalDSpyLM works with dspy.Predict"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM()

        # Verify it has the required interface for DSPy
        assert hasattr(lm, 'basic_request')
        assert callable(lm.basic_request)
        assert hasattr(lm, '__call__')
        assert callable(lm)

    def test_fractal_dspy_lm_model_attribute(self, mock_configure_lm, mock_unified_lm):
        """Test FractalDSpyLM has model attribute required by DSPy"""
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM(tier="balanced")

        # DSPy requires model attribute
        assert hasattr(lm, 'model')
        assert lm.model == "fractal-balanced"

    def test_basic_request_with_extra_kwargs(self, mock_configure_lm, mock_unified_lm):
        """Test basic_request passes extra kwargs to unified_lm"""
        mock_response = {
            'text': "Test response",
            'provider': "anthropic",
            'model': "claude-sonnet-4.5",
            'tokens_used': 50
        }
        mock_unified_lm.return_value = mock_response
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM(tier="balanced")

        # Call with extra kwargs
        result = lm.basic_request(
            prompt="Test prompt",
            custom_param="custom_value",
            another_param=123
        )

        # Verify unified_lm was called with extra kwargs
        call_kwargs = mock_unified_lm.call_args[1]
        assert "custom_param" in call_kwargs
        assert call_kwargs["custom_param"] == "custom_value"
        assert "another_param" in call_kwargs
        assert call_kwargs["another_param"] == 123

    def test_basic_request_error_handling(self, mock_configure_lm, mock_unified_lm):
        """Test basic_request handles errors and re-raises"""
        mock_unified_lm.side_effect = Exception("API Error")
        mock_configure_lm.return_value = mock_unified_lm

        lm = FractalDSpyLM(tier="balanced")

        with pytest.raises(Exception) as exc_info:
            lm.basic_request(prompt="Test")

        assert "API Error" in str(exc_info.value)
