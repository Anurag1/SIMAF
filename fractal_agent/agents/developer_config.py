"""
Developer Agent Configuration

Configuration for the DeveloperAgent - a language-agnostic operational agent
that generates production-quality code by learning from existing patterns,
research, and project conventions.

Features:
- Language-agnostic with pluggable language modules
- Quick mode (direct generation) and thorough mode (multi-agent coordination)
- Learns from: existing codebase, research, GraphRAG, config files
- Validation pipeline integration

Author: BMad
Date: 2025-10-19
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
from ..utils.model_config import Tier
from ..verification import Goal, Evidence, VerificationResult


@dataclass
class DeveloperConfig:
    """
    Configuration for DeveloperAgent.

    The DeveloperAgent is a System 1 (Operational) agent specialized for
    code generation tasks. It complements ResearchAgent by focusing on
    implementation rather than analysis.

    Attributes:
        tier: Model tier for code generation (default: balanced)
        language: Primary language ("python", "javascript", "markdown", etc.)
        mode: "quick" for direct generation, "thorough" for multi-agent coordination
        max_iterations: Maximum validation/refinement iterations (default: 3)
        learn_from_codebase: Read existing code to learn patterns
        use_research_agent: Delegate to ResearchAgent for best practices
        use_graphrag: Query long-term memory for learned patterns
        follow_config_files: Parse .editorconfig, pyproject.toml, etc.
        validation_enabled: Run validation pipeline (syntax, style, tests, quality)
        auto_fix_style: Automatically fix style issues with linters
        generate_tests: Include test generation
        include_documentation: Include docstrings/comments
    """

    # Model configuration
    tier: Tier = "balanced"

    # Language and mode
    language: str = "python"
    mode: str = "quick"  # "quick" or "thorough"

    # Iteration and refinement
    max_iterations: int = 3

    # Learning sources
    learn_from_codebase: bool = True
    use_research_agent: bool = True
    use_graphrag: bool = True
    follow_config_files: bool = True

    # Validation and quality
    validation_enabled: bool = True
    auto_fix_style: bool = True
    generate_tests: bool = True
    include_documentation: bool = True

    # File writing capabilities (uses Claude Agent SDK tools)
    enable_file_writing: bool = False  # Default: safe (no file operations)
    project_root: Optional[Path] = None  # Working directory for file operations

    # Language-specific settings
    language_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.mode not in ["quick", "thorough"]:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'quick' or 'thorough'")

        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")

        # Set language-specific defaults if not provided
        if not self.language_config and self.language == "python":
            self.language_config = {
                "style_guide": "PEP 8",
                "type_hints": True,
                "docstring_style": "Google",
                "test_framework": "pytest",
                "linters": ["black", "flake8", "mypy"],
                "formatters": ["black"],
                "line_length": 88  # black default
            }
        elif not self.language_config and self.language == "javascript":
            self.language_config = {
                "style_guide": "Airbnb",
                "type_hints": True,  # TypeScript
                "docstring_style": "JSDoc",
                "test_framework": "jest",
                "linters": ["eslint"],
                "formatters": ["prettier"],
                "line_length": 80
            }

    def to_llm_config(self) -> Dict[str, Any]:
        """
        Convert DeveloperConfig to LLM provider configuration.

        Returns configuration dict that enables file writing via Claude Agent SDK.
        This is passed to AnthropicProvider via UnifiedLM.

        Returns:
            Dict with provider-specific config for UnifiedLM
            Format: {"anthropic": {"allowed_tools": [...], "permission_mode": ..., "cwd": ...}}
        """
        if not self.enable_file_writing:
            return {}

        # Build Anthropic-specific config
        anthropic_config = {
            # Enable Read/Write/Edit tools from Claude Agent SDK
            "allowed_tools": ["Read", "Write", "Edit"],

            # Auto-accept file operations (agents are trusted)
            "permission_mode": "acceptEdits"
        }

        # Set working directory
        if self.project_root:
            anthropic_config["cwd"] = str(self.project_root)

        # Return in UnifiedLM's expected format (provider-keyed)
        return {"anthropic": anthropic_config}


@dataclass
class CodeGenerationTask:
    """
    Represents a code generation task with goal-based verification.

    Attributes:
        specification: What to implement (requirements, features, etc.)
        language: Target programming language
        context: Additional context (existing code, examples, etc.)
        output_path: Where to write the generated code (optional)
        test_path: Where to write the generated tests (optional)
        expected_artifacts: List of file paths that MUST be created (for verification)
        success_criteria: List of criteria that must be met (for goal verification)
        goal: Optional pre-constructed Goal object (overrides auto-generation)
    """

    specification: str
    language: str = "python"
    context: Optional[str] = None
    output_path: Optional[str] = None
    test_path: Optional[str] = None
    expected_artifacts: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    goal: Optional[Goal] = None

    def __post_init__(self):
        """Auto-populate expected_artifacts and success_criteria if not provided."""
        # Auto-populate expected_artifacts from paths
        if not self.expected_artifacts:
            if self.output_path:
                self.expected_artifacts.append(self.output_path)
            if self.test_path:
                self.expected_artifacts.append(self.test_path)

        # Auto-populate success_criteria if not provided
        if not self.success_criteria and (self.output_path or self.expected_artifacts):
            self.success_criteria = [
                f"Code is syntactically valid {self.language}",
                "Code implements the specification requirements"
            ]
            if self.output_path:
                self.success_criteria.insert(0, f"Source code file exists at {self.output_path}")
            if self.test_path:
                self.success_criteria.append(f"Test file exists at {self.test_path}")
                self.success_criteria.append("Tests are executable")

    def to_goal(self) -> Goal:
        """
        Convert task to a Goal for verification.

        Returns:
            Goal object configured for this code generation task
        """
        if self.goal:
            return self.goal

        from ..verification import create_code_generation_goal

        return create_code_generation_goal(
            specification=self.specification,
            output_path=self.output_path or (self.expected_artifacts[0] if self.expected_artifacts else "code.py"),
            test_path=self.test_path,
            language=self.language
        )


@dataclass
class CodeGenerationResult:
    """
    Result from code generation with goal-based verification.

    Attributes:
        code: Generated source code
        tests: Generated test code
        documentation: Generated documentation
        goal_achieved: Whether the goal was ACHIEVED (not just attempted)
        verification_result: Detailed VerificationResult from goal verification
        evidence: Evidence collected (files created, tests run, etc.)
        iterations: Number of refinement iterations
        metadata: Additional metadata (tokens, cost, etc.)

    Backward Compatibility:
        validation_passed: Alias for goal_achieved (deprecated)
        validation_details: Alias for verification_result.to_dict() (deprecated)
    """

    code: str
    tests: Optional[str] = None
    documentation: Optional[str] = None
    goal_achieved: bool = False
    verification_result: Optional[VerificationResult] = None
    evidence: Optional[Evidence] = None
    iterations: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Backward compatibility properties
    @property
    def validation_passed(self) -> bool:
        """Backward compatibility: alias for goal_achieved."""
        return self.goal_achieved

    @property
    def validation_details(self) -> Dict[str, Any]:
        """Backward compatibility: alias for verification_result."""
        if self.verification_result:
            return self.verification_result.to_dict()
        return {}


# ============================================================================
# Preset Configurations
# ============================================================================

class PresetDeveloperConfigs:
    """Preset configurations for common development scenarios."""

    @staticmethod
    def quick_python() -> DeveloperConfig:
        """
        Quick Python code generation.

        - Direct generation (no multi-agent coordination)
        - Basic validation only
        - Fast iteration
        """
        return DeveloperConfig(
            tier="cheap",
            language="python",
            mode="quick",
            max_iterations=2,
            use_research_agent=False,  # Skip research for speed
            use_graphrag=True,  # Still check learned patterns
            validation_enabled=True,
            auto_fix_style=True
        )

    @staticmethod
    def thorough_python() -> DeveloperConfig:
        """
        Thorough Python code generation.

        - Multi-agent coordination
        - Research best practices
        - Comprehensive validation
        - Multiple refinement iterations
        """
        return DeveloperConfig(
            tier="balanced",
            language="python",
            mode="thorough",
            max_iterations=3,
            use_research_agent=True,  # Research best practices
            use_graphrag=True,
            learn_from_codebase=True,
            validation_enabled=True,
            auto_fix_style=True,
            generate_tests=True,
            include_documentation=True
        )

    @staticmethod
    def production_python() -> DeveloperConfig:
        """
        Production-ready Python code generation.

        - Highest quality model tier
        - All validation and learning enabled
        - Comprehensive testing
        - Full documentation
        """
        return DeveloperConfig(
            tier="expensive",
            language="python",
            mode="thorough",
            max_iterations=5,  # More refinement for production
            learn_from_codebase=True,
            use_research_agent=True,
            use_graphrag=True,
            follow_config_files=True,
            validation_enabled=True,
            auto_fix_style=True,
            generate_tests=True,
            include_documentation=True
        )

    @staticmethod
    def javascript_quick() -> DeveloperConfig:
        """Quick JavaScript/TypeScript code generation."""
        return DeveloperConfig(
            tier="cheap",
            language="javascript",
            mode="quick",
            max_iterations=2,
            use_research_agent=False,
            validation_enabled=True,
            auto_fix_style=True
        )

    @staticmethod
    def markdown_docs() -> DeveloperConfig:
        """Markdown documentation generation."""
        return DeveloperConfig(
            tier="cheap",
            language="markdown",
            mode="quick",
            max_iterations=1,
            use_research_agent=False,
            validation_enabled=False,  # No linters for markdown
            generate_tests=False,  # No tests for docs
            include_documentation=False  # Docs ARE the output
        )
