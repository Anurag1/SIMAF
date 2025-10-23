"""
Workflows for Fractal Agent Ecosystem

This module contains LangGraph workflow implementations for Phase 0+.

Available Workflows:
- research_workflow: Linear research workflow (Phase 0)
- intelligence_workflow: Conditional intelligence analysis workflow (Phase 3)

Author: BMad
Date: 2025-10-19
"""

from .research_workflow import (
    create_research_workflow,
    run_research_workflow
)
from .intelligence_workflow import (
    create_intelligence_workflow,
    run_intelligence_workflow
)

__all__ = [
    "create_research_workflow",
    "run_research_workflow",
    "create_intelligence_workflow",
    "run_intelligence_workflow"
]
