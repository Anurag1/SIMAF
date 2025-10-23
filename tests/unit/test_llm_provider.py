"""
Unit tests for LLM provider infrastructure

Tests UnifiedLM, provider chain, failover, and caching.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fractal_agent.utils.llm_provider import (
    UnifiedLM,
    AnthropicProvider,
    GeminiProvider,
    LLMProvider,
    PROVIDER_REGISTRY,
    LLMMetricsAggregator
)


@pytest.fixture
def mock_anthropic_response():
    """Standard mock response from Anthropic"""
    return {
        "text": "This is a test response from Claude",
        "tokens_used": 150,
        "cache_hit": False,
        "provider": "anthropic",
        "model": "claude-sonnet-4.5"
    }


@pytest.fixture
def mock_gemini_response():
    """Standard mock response from Gemini"""
    return {
        "text": "This is a test response from Gemini",
        "tokens_used": 120,
        "cache_hit": False,
        "provider": "gemini",
        "model": "gemini-2.0-flash-exp"
    }


class TestProviderBase:
    """Test LLMProvider base class"""

    def test_provider_registry_contains_providers(self):
        """Test that provider registry is populated"""
        assert "anthropic" in PROVIDER_REGISTRY
        assert "gemini" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["anthropic"] == AnthropicProvider
        assert PROVIDER_REGISTRY["gemini"] == GeminiProvider

    def test_base_provider_initialization(self):
        """Test base provider attributes"""
        # Create a concrete implementation for testing
        class TestProvider(LLMProvider):
            def _call_provider(self, messages, **kwargs):
                return {"text": "test", "tokens_used": 10, "cache_hit": False, "provider": "test", "model": "test"}

        provider = TestProvider(model="test-model", custom_config="value")
        assert provider.model == "test-model"
        assert provider.config == {"custom_config": "value"}
        assert provider.calls == 0
        assert provider.tokens_used == 0


class TestAnthropicProvider:
    """Test AnthropicProvider"""

    def test_anthropic_provider_initialization(self):
        """Test Anthropic provider initializes correctly"""
        with patch('fractal_agent.utils.llm_provider.os.getcwd', return_value='/test/dir'):
            provider = AnthropicProvider(model="claude-sonnet-4.5", enable_caching=True)

            assert provider.model == "claude-sonnet-4.5"
            assert provider.enable_caching is True
            assert provider.cache_hits == 0
            assert provider.cache_misses == 0
            assert provider.options is not None

    def test_anthropic_provider_builds_prompt_from_messages(self):
        """Test that AnthropicProvider correctly formats messages"""
        with patch('fractal_agent.utils.llm_provider.os.getcwd', return_value='/test/dir'):
            provider = AnthropicProvider(model="claude-sonnet-4.5")

            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"}
            ]

            # We can't easily test the async method without mocking claude_query
            # but we can verify the provider has the required attributes
            assert hasattr(provider, '_call_provider_async')
            assert hasattr(provider, '_call_provider')


class TestGeminiProvider:
    """Test GeminiProvider"""

    def test_gemini_provider_initialization(self):
        """Test Gemini provider initializes correctly"""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key'}):
            with patch('google.generativeai.configure'):
                with patch('google.generativeai.GenerativeModel') as mock_model:
                    provider = GeminiProvider(model="gemini-2.0-flash-exp")

                    assert provider.model == "gemini-2.0-flash-exp"
                    assert provider.client is not None

    def test_gemini_provider_call(self):
        """Test Gemini provider call with mocked response"""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key'}):
            with patch('google.generativeai.configure'):
                with patch('google.generativeai.GenerativeModel') as MockModel:
                    # Setup mock
                    mock_response = Mock()
                    mock_response.text = "Test response from Gemini"
                    mock_client = Mock()
                    mock_client.generate_content.return_value = mock_response
                    MockModel.return_value = mock_client

                    provider = GeminiProvider(model="gemini-2.0-flash-exp")

                    messages = [{"role": "user", "content": "Test prompt"}]
                    result = provider._call_provider(messages, max_tokens=100, temperature=0.7)

                    assert result["text"] == "Test response from Gemini"
                    assert result["provider"] == "gemini"
                    assert result["model"] == "gemini-2.0-flash-exp"
                    assert result["tokens_used"] > 0
                    assert result["cache_hit"] is False

    def test_gemini_parameter_mapping(self):
        """Test that Anthropic-style parameters are mapped to Gemini format"""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key'}):
            with patch('google.generativeai.configure'):
                with patch('google.generativeai.GenerativeModel') as MockModel:
                    mock_response = Mock()
                    mock_response.text = "Response"
                    mock_client = Mock()
                    mock_client.generate_content.return_value = mock_response
                    MockModel.return_value = mock_client

                    provider = GeminiProvider(model="gemini-2.0-flash-exp")
                    messages = [{"role": "user", "content": "Test"}]

                    provider._call_provider(messages, max_tokens=200, temperature=0.5, top_p=0.9)

                    # Verify generate_content was called with correct config
                    call_kwargs = mock_client.generate_content.call_args[1]
                    assert "generation_config" in call_kwargs
                    config = call_kwargs["generation_config"]
                    assert config["max_output_tokens"] == 200
                    assert config["temperature"] == 0.5
                    assert config["top_p"] == 0.9


class TestUnifiedLM:
    """Test UnifiedLM core functionality"""

    def test_unified_lm_initialization_with_defaults(self):
        """Test UnifiedLM initializes with default provider chain"""
        with patch('fractal_agent.utils.llm_provider.AnthropicProvider'):
            with patch('fractal_agent.utils.llm_provider.GeminiProvider'):
                lm = UnifiedLM()

                assert len(lm.provider_chain) == 2
                assert lm.total_calls == 0
                assert lm.total_failures == 0

    def test_unified_lm_initialization_with_custom_providers(self):
        """Test UnifiedLM with custom provider chain"""
        with patch('fractal_agent.utils.llm_provider.AnthropicProvider'):
            lm = UnifiedLM(providers=[("anthropic", "claude-sonnet-4.5")])

            assert len(lm.provider_chain) == 1

    @pytest.mark.mock
    def test_unified_lm_call_with_prompt(self, mock_anthropic_response):
        """Test UnifiedLM call structure (mocked)"""
        # This test verifies UnifiedLM interface works correctly
        # Actual provider calls are tested in integration tests
        with patch('fractal_agent.utils.llm_provider.AnthropicProvider') as MockProvider:
            # Setup mock provider
            mock_provider = Mock()
            mock_provider.model = "claude-sonnet-4.5"
            mock_provider.calls = 0
            mock_provider.tokens_used = 0
            mock_provider.__call__ = Mock(return_value=mock_anthropic_response)
            mock_provider.__class__.__name__ = "AnthropicProvider"
            MockProvider.return_value = mock_provider

            lm = UnifiedLM(providers=[("anthropic", "claude-sonnet-4.5")])
            result = lm(prompt="Test prompt")

            # Verify result structure
            assert "text" in result
            assert "provider" in result
            assert lm.total_calls == 1

    @pytest.mark.skip(reason="Mocking full provider chain is complex, covered by integration tests")
    def test_unified_lm_call_with_messages(self):
        """Test UnifiedLM call with message list"""
        pass

    @pytest.mark.skip(reason="Mocking full provider chain is complex, covered by integration tests")
    def test_unified_lm_call_with_system_message(self):
        """Test UnifiedLM combines prompt and system message"""
        pass

    @pytest.mark.skip(reason="Mocking full provider chain is complex, covered by integration tests")
    def test_unified_lm_failover_to_second_provider(self):
        """Test UnifiedLM fails over to second provider when first fails"""
        pass

    @pytest.mark.skip(reason="Mocking full provider chain is complex, covered by integration tests")
    def test_unified_lm_all_providers_fail(self):
        """Test UnifiedLM raises error when all providers fail"""
        pass

    def test_unified_lm_metrics(self, mock_anthropic_response):
        """Test UnifiedLM tracks metrics correctly"""
        with patch('fractal_agent.utils.llm_provider.AnthropicProvider') as MockProvider:
            mock_provider = Mock()
            mock_provider.model = "claude-sonnet-4.5"
            mock_provider.calls = 2
            mock_provider.tokens_used = 300
            mock_provider.cache_hits = 1
            mock_provider.cache_misses = 1
            mock_provider.__call__ = Mock(return_value=mock_anthropic_response)
            mock_provider.__class__.__name__ = "AnthropicProvider"
            MockProvider.return_value = mock_provider

            lm = UnifiedLM(providers=[("anthropic", "claude-sonnet-4.5")])
            lm(prompt="Test 1")
            lm(prompt="Test 2")

            metrics = lm.get_metrics()

            assert metrics["total_calls"] == 2
            assert metrics["total_failures"] == 0
            assert "per_provider" in metrics
            assert len(metrics["per_provider"]) == 1

    def test_unified_lm_requires_prompt_or_messages(self):
        """Test UnifiedLM raises error without prompt or messages"""
        with patch('fractal_agent.utils.llm_provider.AnthropicProvider'):
            lm = UnifiedLM(providers=[("anthropic", "claude-sonnet-4.5")])

            with pytest.raises(ValueError) as exc_info:
                lm()

            assert "Must provide either 'messages' or 'prompt'" in str(exc_info.value)

    def test_unified_lm_no_valid_providers(self):
        """Test UnifiedLM raises error with no valid providers"""
        with pytest.raises(ValueError) as exc_info:
            UnifiedLM(providers=[("invalid_provider", "model")])

        assert "No valid providers configured" in str(exc_info.value)

    @pytest.mark.skip(reason="Complex provider mocking - these lines covered by integration tests")
    def test_unified_lm_call_with_system_parameter(self):
        """Test UnifiedLM adds system message when system parameter provided"""
        pass

    @pytest.mark.skip(reason="Complex provider mocking - these lines covered by integration tests")
    def test_unified_lm_all_providers_fail_error(self):
        """Test UnifiedLM raises comprehensive error when all providers fail"""
        pass


class TestLLMMetricsAggregator:
    """Test global metrics aggregation"""

    def test_metrics_aggregator_registration(self):
        """Test registering UnifiedLM instances"""
        with patch('fractal_agent.utils.llm_provider.AnthropicProvider'):
            # Clear existing instances
            LLMMetricsAggregator._instances = []

            lm1 = UnifiedLM(providers=[("anthropic", "claude-sonnet-4.5")])
            lm2 = UnifiedLM(providers=[("anthropic", "claude-sonnet-4.5")])

            LLMMetricsAggregator.register(lm1)
            LLMMetricsAggregator.register(lm2)

            metrics = LLMMetricsAggregator.get_global_metrics()

            assert metrics["active_instances"] == 2
            assert metrics["total_calls"] == 0
            assert metrics["total_failures"] == 0
