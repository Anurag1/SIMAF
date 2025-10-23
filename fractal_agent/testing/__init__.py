"""
Testing Framework - Phase 3

A/B testing framework for comparing agent variants and MIPRO optimization strategies.

Author: BMad
Date: 2025-10-19
"""

from fractal_agent.testing.ab_testing import (
    ABTestFramework,
    Variant,
    VariantType,
    ABTestResult,
    quick_ab_test
)

from fractal_agent.testing.mipro_ab_testing import (
    run_mipro_ab_test,
    compare_mipro_presets
)

__all__ = [
    # A/B Testing Core
    "ABTestFramework",
    "Variant",
    "VariantType",
    "ABTestResult",
    "quick_ab_test",

    # MIPRO Integration
    "run_mipro_ab_test",
    "compare_mipro_presets",
]
