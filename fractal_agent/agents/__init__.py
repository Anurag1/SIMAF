"""
Fractal Agent Ecosystem - Agents

This package contains all operational agents in the system.
Each agent follows VSM principles and uses the unified LLM infrastructure.

Available Agents:
- ResearchAgent: Multi-stage research with synthesis and validation (System 1)
- DeveloperAgent: Code generation with goal-based verification (System 1)
- CoordinationAgent: Multi-agent orchestration and conflict resolution (System 2)
- IntelligenceAgent: Performance reflection and learning (System 4)
- PolicyAgent: Ethical governance and strategic direction (System 5)
- KnowledgeExtractionAgent: Automatic knowledge extraction from task outputs (System 1)

Author: BMad
Date: 2025-10-18
"""

from .research_agent import ResearchAgent
from .developer_agent import DeveloperAgent
from .coordination_agent import CoordinationAgent
from .intelligence_agent import IntelligenceAgent
from .policy_agent import PolicyAgent
from .knowledge_extraction_agent import KnowledgeExtractionAgent

from .developer_config import DeveloperConfig, PresetDeveloperConfigs
from .intelligence_config import IntelligenceConfig, PresetIntelligenceConfigs
from .policy_config import PolicyConfig, PresetPolicyConfigs

__all__ = [
    "ResearchAgent",
    "DeveloperAgent",
    "CoordinationAgent",
    "IntelligenceAgent",
    "PolicyAgent",
    "KnowledgeExtractionAgent",
    "DeveloperConfig",
    "PresetDeveloperConfigs",
    "IntelligenceConfig",
    "PresetIntelligenceConfigs",
    "PolicyConfig",
    "PresetPolicyConfigs"
]
