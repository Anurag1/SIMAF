"""
Model Registry for Fractal Agent Ecosystem

Provides dynamic model discovery and tier-based selection instead of hardcoded model names.

Architecture:
- Fetches available models from provider APIs
- Caches model information for 24 hours
- Supports tier-based selection (cheap/balanced/expensive/premium)
- YAML configuration for pricing and tier assignments
- Extensible for new providers

Author: BMad
Date: 2025-10-18
"""

import os
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Literal
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

# Try to import provider SDKs
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

# Type definitions
Tier = Literal["cheap", "balanced", "expensive", "premium"]
Provider = Literal["anthropic", "gemini", "openai", "together"]


@dataclass
class ModelInfo:
    """
    Information about a specific model.

    Attributes:
        model_id: Unique model identifier (e.g., "claude-sonnet-4.5")
        provider: Provider name (e.g., "anthropic")
        tier: Cost/capability tier
        input_cost_per_mtok: Cost per million input tokens (USD)
        output_cost_per_mtok: Cost per million output tokens (USD)
        context_window: Maximum context window size
        supports_vision: Whether model supports image inputs
        supports_function_calling: Whether model supports function/tool calling
        supports_caching: Whether model supports prompt caching
        description: Human-readable description
        last_updated: When this info was last fetched
    """
    model_id: str
    provider: Provider
    tier: Tier
    input_cost_per_mtok: float
    output_cost_per_mtok: float
    context_window: int
    supports_vision: bool = False
    supports_function_calling: bool = True
    supports_caching: bool = False
    description: str = ""
    last_updated: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        """Create ModelInfo from dictionary"""
        return cls(**data)


class ModelRegistry:
    """
    Central registry for all LLM models across providers.

    Provides:
    - Dynamic model discovery from provider APIs
    - Tier-based model selection
    - Cost and capability information
    - Automatic caching with TTL

    Usage:
        registry = ModelRegistry()
        registry.refresh()  # Fetch latest models

        # Get model by tier
        cheap_model = registry.get_model_by_tier("cheap", provider="anthropic")
        premium_model = registry.get_model_by_tier("premium")

        # Get all models for a provider
        anthropic_models = registry.get_models_by_provider("anthropic")

        # Get model info
        info = registry.get_model_info("claude-sonnet-4.5")
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        cache_path: Optional[str] = None,
        cache_ttl_hours: int = 24
    ):
        """
        Initialize ModelRegistry.

        Args:
            config_path: Path to YAML config file with pricing/tier info
            cache_path: Path to JSON cache file
            cache_ttl_hours: How long to cache model info (default 24 hours)
        """
        # Default paths
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                "../../config/models_pricing.yaml"
            )

        if cache_path is None:
            cache_path = os.path.join(
                os.path.dirname(__file__),
                "../../cache/models_cache.json"
            )

        self.config_path = Path(config_path)
        self.cache_path = Path(cache_path)
        self.cache_ttl_hours = cache_ttl_hours

        # In-memory model registry
        self.models: Dict[str, ModelInfo] = {}

        # Load configuration
        self.config = self._load_config()

        # Load cache (if valid)
        self._load_cache()

        # Auto-refresh if cache is stale
        if self._is_cache_stale():
            logger.info("Cache is stale, refreshing models...")
            self.refresh()

    def _load_config(self) -> Dict[str, Any]:
        """Load pricing and tier configuration from YAML"""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return self._get_default_config()

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded model config from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if YAML not available"""
        return {
            "models": {
                "anthropic": {
                    "claude-sonnet-4.5": {
                        "tier": "balanced",
                        "input_cost": 3.0,
                        "output_cost": 15.0,
                        "context_window": 200000,
                        "supports_vision": True,
                        "supports_caching": True
                    },
                    "claude-opus-4.1": {
                        "tier": "premium",
                        "input_cost": 15.0,
                        "output_cost": 75.0,
                        "context_window": 200000,
                        "supports_vision": True,
                        "supports_caching": True
                    },
                    "claude-haiku-4.5": {
                        "tier": "cheap",
                        "input_cost": 1.0,
                        "output_cost": 5.0,
                        "context_window": 200000,
                        "supports_vision": True,
                        "supports_caching": True
                    }
                },
                "gemini": {
                    "gemini-2.0-flash-exp": {
                        "tier": "cheap",
                        "input_cost": 0.0,  # Free tier
                        "output_cost": 0.0,
                        "context_window": 1000000,
                        "supports_vision": True,
                        "supports_caching": False
                    },
                    "gemini-1.5-pro": {
                        "tier": "balanced",
                        "input_cost": 1.25,
                        "output_cost": 5.0,
                        "context_window": 2000000,
                        "supports_vision": True,
                        "supports_caching": True
                    }
                }
            }
        }

    def _load_cache(self) -> None:
        """Load models from cache if available and not expired"""
        if not self.cache_path.exists():
            logger.info("No cache file found, will fetch fresh models")
            return

        try:
            with open(self.cache_path, 'r') as f:
                cache_data = json.load(f)

            # Check cache timestamp
            cache_time = datetime.fromisoformat(cache_data.get("timestamp", "2000-01-01"))
            if datetime.now() - cache_time > timedelta(hours=self.cache_ttl_hours):
                logger.info("Cache expired, will fetch fresh models")
                return

            # Load models from cache
            for model_id, model_data in cache_data.get("models", {}).items():
                self.models[model_id] = ModelInfo.from_dict(model_data)

            logger.info(f"Loaded {len(self.models)} models from cache")

        except Exception as e:
            logger.error(f"Error loading cache: {e}")

    def _save_cache(self) -> None:
        """Save current models to cache"""
        try:
            # Ensure cache directory exists
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "models": {
                    model_id: model.to_dict()
                    for model_id, model in self.models.items()
                }
            }

            with open(self.cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)

            logger.info(f"Saved {len(self.models)} models to cache")

        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _is_cache_stale(self) -> bool:
        """Check if cache needs refresh"""
        if not self.cache_path.exists():
            return True

        try:
            with open(self.cache_path, 'r') as f:
                cache_data = json.load(f)
                cache_time = datetime.fromisoformat(cache_data.get("timestamp", "2000-01-01"))
                return datetime.now() - cache_time > timedelta(hours=self.cache_ttl_hours)
        except:
            return True

    def refresh(self) -> None:
        """
        Refresh model information from all providers.

        This method:
        1. Fetches available models from provider APIs
        2. Merges with configuration data
        3. Updates in-memory registry
        4. Saves to cache
        """
        logger.info("Refreshing model registry...")

        # Clear current models
        self.models = {}

        # Fetch from each provider
        if ANTHROPIC_AVAILABLE:
            self._fetch_anthropic_models()

        if GEMINI_AVAILABLE:
            self._fetch_gemini_models()

        # Save to cache
        self._save_cache()

        logger.info(f"Registry refreshed with {len(self.models)} models")

    def _fetch_anthropic_models(self) -> None:
        """Fetch Anthropic models from configuration"""
        # Anthropic doesn't have a public API to list models dynamically
        # So we use the configuration file as source of truth

        config_models = self.config.get("models", {}).get("anthropic", {})

        for model_id, model_data in config_models.items():
            try:
                model_info = ModelInfo(
                    model_id=model_id,
                    provider="anthropic",
                    tier=model_data.get("tier", "balanced"),
                    input_cost_per_mtok=model_data.get("input_cost", 0.0),
                    output_cost_per_mtok=model_data.get("output_cost", 0.0),
                    context_window=model_data.get("context_window", 200000),
                    supports_vision=model_data.get("supports_vision", False),
                    supports_function_calling=model_data.get("supports_function_calling", True),
                    supports_caching=model_data.get("supports_caching", False),
                    description=model_data.get("description", ""),
                    last_updated=datetime.now().isoformat()
                )
                self.models[model_id] = model_info
                logger.debug(f"Added Anthropic model: {model_id}")

            except Exception as e:
                logger.error(f"Error adding Anthropic model {model_id}: {e}")

    def _fetch_gemini_models(self) -> None:
        """Fetch Gemini models from configuration"""
        # Gemini has a models API, but for now use configuration
        # TODO: Implement dynamic fetching from genai.list_models()

        config_models = self.config.get("models", {}).get("gemini", {})

        for model_id, model_data in config_models.items():
            try:
                model_info = ModelInfo(
                    model_id=model_id,
                    provider="gemini",
                    tier=model_data.get("tier", "balanced"),
                    input_cost_per_mtok=model_data.get("input_cost", 0.0),
                    output_cost_per_mtok=model_data.get("output_cost", 0.0),
                    context_window=model_data.get("context_window", 1000000),
                    supports_vision=model_data.get("supports_vision", False),
                    supports_function_calling=model_data.get("supports_function_calling", True),
                    supports_caching=model_data.get("supports_caching", False),
                    description=model_data.get("description", ""),
                    last_updated=datetime.now().isoformat()
                )
                self.models[model_id] = model_info
                logger.debug(f"Added Gemini model: {model_id}")

            except Exception as e:
                logger.error(f"Error adding Gemini model {model_id}: {e}")

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get detailed information about a specific model.

        Args:
            model_id: Model identifier (e.g., "claude-sonnet-4.5")

        Returns:
            ModelInfo object or None if not found
        """
        return self.models.get(model_id)

    def get_models_by_provider(self, provider: Provider) -> List[ModelInfo]:
        """
        Get all models for a specific provider.

        Args:
            provider: Provider name (e.g., "anthropic")

        Returns:
            List of ModelInfo objects
        """
        return [
            model for model in self.models.values()
            if model.provider == provider
        ]

    def get_models_by_tier(self, tier: Tier) -> List[ModelInfo]:
        """
        Get all models in a specific tier.

        Args:
            tier: Tier name (cheap/balanced/expensive/premium)

        Returns:
            List of ModelInfo objects
        """
        return [
            model for model in self.models.values()
            if model.tier == tier
        ]

    def get_model_by_tier(
        self,
        tier: Tier,
        provider: Optional[Provider] = None,
        require_vision: bool = False,
        require_caching: bool = False
    ) -> Optional[ModelInfo]:
        """
        Get a model matching tier and capability requirements.

        Args:
            tier: Desired tier (cheap/balanced/expensive/premium)
            provider: Optional provider filter
            require_vision: Require vision support
            require_caching: Require caching support

        Returns:
            Best matching ModelInfo or None if no match
        """
        # Filter by tier
        candidates = self.get_models_by_tier(tier)

        # Filter by provider if specified
        if provider:
            candidates = [m for m in candidates if m.provider == provider]

        # Filter by capabilities
        if require_vision:
            candidates = [m for m in candidates if m.supports_vision]

        if require_caching:
            candidates = [m for m in candidates if m.supports_caching]

        # Return first match (sorted by input cost)
        if candidates:
            candidates.sort(key=lambda m: m.input_cost_per_mtok)
            return candidates[0]

        return None

    def list_all_models(self) -> Dict[str, ModelInfo]:
        """Get all registered models"""
        return self.models.copy()

    def get_tier_summary(self) -> Dict[Tier, int]:
        """Get count of models per tier"""
        summary = {"cheap": 0, "balanced": 0, "expensive": 0, "premium": 0}
        for model in self.models.values():
            summary[model.tier] += 1
        return summary

    def get_provider_summary(self) -> Dict[Provider, int]:
        """Get count of models per provider"""
        summary: Dict[Provider, int] = {}
        for model in self.models.values():
            summary[model.provider] = summary.get(model.provider, 0) + 1
        return summary


# Global singleton instance
_global_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get or create global ModelRegistry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry


# Quick test
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("ModelRegistry Test")
    print("=" * 80)
    print()

    # Create registry
    registry = ModelRegistry()

    # Refresh (loads from config)
    registry.refresh()

    # Summary
    print(f"Total models: {len(registry.models)}")
    print(f"By tier: {registry.get_tier_summary()}")
    print(f"By provider: {registry.get_provider_summary()}")
    print()

    # Get models by tier
    for tier in ["cheap", "balanced", "expensive", "premium"]:
        model = registry.get_model_by_tier(tier)
        if model:
            print(f"{tier.upper():10} → {model.model_id:30} "
                  f"(${model.input_cost_per_mtok}/${model.output_cost_per_mtok} per Mtok)")

    print()

    # Get provider models
    print("Anthropic models:")
    for model in registry.get_models_by_provider("anthropic"):
        print(f"  - {model.model_id} ({model.tier})")

    print()
    print("Gemini models:")
    for model in registry.get_models_by_provider("gemini"):
        print(f"  - {model.model_id} ({model.tier})")

    print()
    print("=" * 80)
    print("✓ ModelRegistry test complete!")
    print("=" * 80)
