"""
Unit tests for model registry

Tests model loading, tier lookup, and registry functionality.
"""

import pytest
from fractal_agent.utils.model_registry import ModelRegistry, get_registry


class TestModelRegistry:
    """Test ModelRegistry functionality"""

    def test_registry_initialization(self):
        """Test registry initializes with default models"""
        registry = get_registry()

        assert registry is not None
        assert hasattr(registry, 'models')
        assert len(registry.models) > 0

    def test_get_model_by_tier_cheap(self):
        """Test getting cheap tier model"""
        registry = get_registry()
        model = registry.get_model_by_tier("cheap")

        assert model is not None
        assert hasattr(model, 'provider')
        assert hasattr(model, 'model_id')

    def test_get_model_by_tier_balanced(self):
        """Test getting balanced tier model"""
        registry = get_registry()
        model = registry.get_model_by_tier("balanced")

        assert model is not None
        assert model.provider in ["anthropic", "gemini"]

    def test_get_model_by_tier_expensive(self):
        """Test getting expensive tier model"""
        registry = get_registry()
        model = registry.get_model_by_tier("expensive")

        assert model is not None
        assert hasattr(model, 'model_id')

    def test_get_model_by_tier_premium(self):
        """Test getting premium tier model"""
        registry = get_registry()
        model = registry.get_model_by_tier("premium")

        assert model is not None
        assert hasattr(model, 'provider')

    def test_get_models_by_tier(self):
        """Test getting all models for a tier"""
        registry = get_registry()
        models = registry.get_models_by_tier("balanced")

        assert models is not None
        assert len(models) > 0

    def test_get_models_by_provider(self):
        """Test getting models by provider"""
        registry = get_registry()
        anthropic_models = registry.get_models_by_provider("anthropic")
        gemini_models = registry.get_models_by_provider("gemini")

        assert len(anthropic_models) > 0
        assert len(gemini_models) > 0

    def test_get_tier_summary(self):
        """Test getting tier summary"""
        registry = get_registry()
        summary = registry.get_tier_summary()

        assert summary is not None
        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_get_provider_summary(self):
        """Test getting provider summary"""
        registry = get_registry()
        summary = registry.get_provider_summary()

        assert summary is not None
        assert isinstance(summary, dict)
        assert "anthropic" in summary
        assert "gemini" in summary

    def test_get_model_info(self):
        """Test getting specific model info"""
        registry = get_registry()
        # Get a model first
        model = registry.get_model_by_tier("balanced")

        # Look it up by ID
        info = registry.get_model_info(model.model_id)

        assert info is not None
        assert info.model_id == model.model_id
