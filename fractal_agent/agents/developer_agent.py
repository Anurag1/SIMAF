"""
Developer Agent - System 1 (Operational)

Language-agnostic code generation agent that learns from existing patterns,
research, and project conventions to generate production-quality code.

Complements ResearchAgent by focusing on implementation rather than analysis.

Author: BMad
Date: 2025-10-19
"""

import dspy
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .developer_config import (
    DeveloperConfig,
    CodeGenerationTask,
    CodeGenerationResult,
    PresetDeveloperConfigs
)
from ..utils.dspy_integration import FractalDSpyLM, configure_dspy
from ..utils.model_config import Tier
from ..verification import (
    Goal,
    Evidence,
    VerificationResult,
    verify_goal,
    EvidenceCollector,
    create_code_generation_goal
)
from ..observability import (
    get_correlation_id, get_tracer, get_logger,
    get_event_store, VSMEvent, set_span_attributes
)

# Use observability-aware structured logger
logger = get_logger(__name__)


# ============================================================================
# DSPy Signatures for Code Generation
# ============================================================================

class CodeGeneration(dspy.Signature):
    """
    Generate production-quality code from specification.

    Input: Specification + context (examples, patterns, requirements)
    Output: Complete, executable, well-documented code

    This signature is optimized for CODE GENERATION, not analysis.
    It should output actual runnable code, not descriptions of code.

    If Write tool is available, use it to save generated code to appropriate files.
    """

    specification = dspy.InputField(
        desc="What to implement: requirements, features, API, etc."
    )
    language = dspy.InputField(
        desc="Programming language (python, javascript, etc.)"
    )
    context = dspy.InputField(
        desc="Example code patterns, existing implementations to follow"
    )
    style_guide = dspy.InputField(
        desc="Style guide to follow (PEP 8, Airbnb, etc.)"
    )
    domain_knowledge = dspy.InputField(
        desc="Relevant domain knowledge and architectural patterns from past implementations"
    )
    recent_examples = dspy.InputField(
        desc="Recent similar code generation tasks and their solutions"
    )
    constraints = dspy.InputField(
        desc="Best practices, coding standards, and project-specific constraints"
    )

    code = dspy.OutputField(
        desc="Complete, executable source code with imports, docstrings, type hints. OUTPUT ONLY THE RAW SOURCE CODE - do NOT wrap in markdown code fences (```), do NOT include explanatory text, do NOT add any formatting. The output must be directly parseable by the language interpreter/compiler."
    )


class TestGeneration(dspy.Signature):
    """
    Generate comprehensive tests for code.

    Input: Code + specification
    Output: Test suite with good coverage

    If Write tool is available, use it to save generated tests to appropriate files.
    """

    code = dspy.InputField(desc="Source code to test")
    specification = dspy.InputField(desc="Original requirements")
    test_framework = dspy.InputField(desc="Testing framework (pytest, jest, etc.)")

    tests = dspy.OutputField(
        desc="Complete test suite with multiple test cases, edge cases, fixtures. OUTPUT ONLY THE RAW TEST CODE - do NOT wrap in markdown code fences (```), do NOT include explanatory text. The output must be directly executable by the test framework."
    )


class CodeRefinement(dspy.Signature):
    """
    Refine code based on validation feedback.

    Input: Original code + validation errors/suggestions
    Output: Improved code that addresses issues
    """

    original_code = dspy.InputField(desc="Code that failed validation")
    validation_feedback = dspy.InputField(
        desc="Errors, warnings, and suggestions from validators"
    )
    specification = dspy.InputField(desc="Original requirements (for reference)")

    refined_code = dspy.OutputField(
        desc="Improved code that fixes validation issues while maintaining functionality. OUTPUT ONLY THE RAW SOURCE CODE - do NOT wrap in markdown code fences (```), do NOT include explanatory text. The output must be directly parseable by the language interpreter/compiler."
    )


# ============================================================================
# Developer Agent
# ============================================================================

class DeveloperAgent(dspy.Module):
    """
    System 1 (Operational) - Code Generation Agent

    Language-agnostic agent that generates production-quality code by:
    1. Learning from existing codebase patterns
    2. Researching best practices (via ResearchAgent delegation)
    3. Querying GraphRAG for learned patterns
    4. Following project config files
    5. Iteratively refining based on validation feedback

    Usage:
        >>> config = PresetDeveloperConfigs.thorough_python()
        >>> agent = DeveloperAgent(config=config)
        >>>
        >>> task = CodeGenerationTask(
        ...     specification="Create ObsidianVault class with Git sync",
        ...     language="python",
        ...     context=read_existing_code("obsidian_export.py")
        ... )
        >>>
        >>> result = agent(task)
        >>> print(result.code)  # Generated code
        >>> print(result.tests)  # Generated tests
    """

    def __init__(
        self,
        config: Optional[DeveloperConfig] = None,
        tier: Optional[Tier] = None,
        graphrag=None,
        memory=None,
        obsidian_vault=None,
        enable_context_prep: bool = True
    ):
        """
        Initialize DeveloperAgent.

        Args:
            config: Agent configuration (default: thorough_python)
            tier: Model tier override (default: from config)
            graphrag: GraphRAG instance for past code patterns (optional)
            memory: ShortTermMemory for recent tasks (optional)
            obsidian_vault: ObsidianVault for coding standards (optional)
            enable_context_prep: Whether to use intelligent context preparation (default: True)
        """
        super().__init__()

        self.config = config or PresetDeveloperConfigs.thorough_python()
        self.tier = tier or self.config.tier
        self.graphrag = graphrag
        self.memory = memory
        self.obsidian_vault = obsidian_vault
        self.enable_context_prep = enable_context_prep

        # Initialize context preparation agent if enabled
        self.context_prep_agent = None
        if self.enable_context_prep:
            try:
                from .context_preparation_agent import ContextPreparationAgent
                self.context_prep_agent = ContextPreparationAgent(
                    graphrag=self.graphrag,
                    memory=self.memory,
                    obsidian_vault=self.obsidian_vault,
                    web_search=None,
                    max_iterations=1,  # Code gen gets 1 iteration (fast)
                    min_confidence=0.6  # Slightly lower threshold for code
                )
                logger.info("Context preparation agent initialized for DeveloperAgent")
            except ImportError:
                logger.warning("ContextPreparationAgent not available, using simple context")
                self.enable_context_prep = False

        # Configure DSPy with the appropriate tier
        # Pass file writing config to LLM provider if enabled
        llm_config = self.config.to_llm_config()
        configure_dspy(tier=self.tier, **llm_config)

        # Define DSPy modules for each stage
        self.generate_code = dspy.ChainOfThought(CodeGeneration)
        self.generate_tests = dspy.ChainOfThought(TestGeneration)
        self.refine_code = dspy.ChainOfThought(CodeRefinement)

        logger.info(
            f"DeveloperAgent initialized: language={self.config.language}, "
            f"mode={self.config.mode}, tier={self.tier}"
        )

    def forward(
        self,
        task: CodeGenerationTask,
        verbose: bool = False
    ) -> CodeGenerationResult:
        """
        Generate code from specification with goal-based verification.

        NEW: This method now uses goal-based verification instead of simple syntax checking.
        The goal is defined by the task (e.g., "create file X with tests at Y"), and we
        verify that the goal was ACHIEVED (not just attempted).

        Args:
            task: Code generation task with specification and context
            verbose: Print progress information

        Returns:
            CodeGenerationResult with generated code, tests, goal achievement status, evidence
        """
        # OBSERVABILITY: Get tracer, event store, and correlation ID
        tracer = get_tracer(__name__)
        event_store = get_event_store()
        correlation_id = get_correlation_id()

        with tracer.start_as_current_span("system1_developer_codegen") as span:
            # OBSERVABILITY: Set span attributes
            set_span_attributes({
                "vsm.tier": "System1_Developer",
                "vsm.operation": "code_generation",
                "task.language": task.language,
                "task.specification": task.specification[:100],
                "correlation_id": correlation_id
            })

            # OBSERVABILITY: Emit code generation started event
            event_store.append(VSMEvent(
                tier="System1_Developer",
                event_type="codegen_started",
                data={
                    "language": task.language,
                    "specification": task.specification,
                    "output_path": task.output_path,
                    "correlation_id": correlation_id
                }
            ))

            logger.info(
                f"System 1 (Developer) code generation started: {task.language}",
                extra={"correlation_id": correlation_id, "language": task.language}
            )

            if verbose:
                print(f"\n{'=' * 80}")
                print(f"DeveloperAgent: Generating {task.language} code")
                print(f"Mode: {self.config.mode}")
                print(f"{'=' * 80}\n")

            # NEW: Create goal from task
            goal = task.to_goal()
            if verbose:
                print(f"Goal: {goal.objective}")
                print(f"Required artifacts: {goal.required_artifacts}")
                print()

            # Stage 1: Prepare context intelligently
            context_package = self._gather_context(task, verbose=verbose)

            # Stage 2: Generate code
            if verbose:
                print(f"[1/3] Generating code... (tier={self.tier})")

            # Format context for code generation
            if context_package:
                formatted_context = context_package.format_for_agent("developer")
                code_result = self.generate_code(
                    specification=task.specification,
                    language=task.language,
                    context=task.context or "No task context provided",
                    style_guide=self.config.language_config.get("style_guide", "Standard"),
                    domain_knowledge=formatted_context["domain_knowledge"],
                    recent_examples=formatted_context["recent_examples"],
                    constraints=formatted_context["constraints"]
                )
            else:
                # Fallback: no intelligent context
                code_result = self.generate_code(
                    specification=task.specification,
                    language=task.language,
                    context=task.context or "No context provided",
                    style_guide=self.config.language_config.get("style_guide", "Standard"),
                    domain_knowledge="No domain knowledge available",
                    recent_examples="No recent examples available",
                    constraints="Follow standard coding best practices"
                )

            current_code = code_result.code
            iterations = 1

            # Write generated code to disk if enabled and path provided
            if task.output_path and current_code:
                self._write_files(current_code, task.output_path, verbose=verbose)

            # OBSERVABILITY: Emit code generated event
            event_store.append(VSMEvent(
                tier="System1_Developer",
                event_type="code_generated",
                data={
                    "language": task.language,
                    "code_length": len(current_code),
                    "output_path": task.output_path,
                    "correlation_id": correlation_id
                }
            ))

            logger.info(
                f"System 1 (Developer) code generated: {len(current_code)} chars",
                extra={"correlation_id": correlation_id, "code_length": len(current_code)}
            )

            # Stage 3: Goal verification and refinement loop
            # NEW: Collect evidence and verify GOAL instead of just checking syntax
            tests = None
            verification_result = None

            if self.config.validation_enabled:
                if verbose:
                    print(f"[2/3] Verifying goal achievement and refining...")

                for iteration in range(self.config.max_iterations):
                    # Generate tests if needed (before verification)
                    if self.config.generate_tests and iteration == 0:
                        test_result = self.generate_tests(
                            code=current_code,
                            specification=task.specification,
                            test_framework=self.config.language_config.get("test_framework", "pytest")
                        )
                        tests = test_result.tests

                        # Write tests immediately so verification can find them
                        if task.test_path and tests:
                            self._write_files(tests, task.test_path, verbose=verbose)

                    # NEW: Collect evidence of what was actually created
                    evidence = EvidenceCollector.collect_for_code_generation(
                        output_path=task.output_path,
                        test_path=task.test_path,
                        language=task.language,
                        metadata={
                            "code_generated": True,
                            "tests_generated": tests is not None,
                            "iteration": iteration + 1
                        }
                    )

                    # Store generated code in evidence metadata
                    evidence.llm_observations = f"Generated code ({len(current_code)} chars)"

                    # NEW: Verify goal achievement (not just syntax)
                    verification_result = self._verify_goal(
                        goal=goal,
                        evidence=evidence,
                        verbose=verbose
                    )

                    if verification_result.is_success:
                        if verbose:
                            print(f"  ✅ Goal achieved on iteration {iteration + 1}")
                        break

                    if iteration < self.config.max_iterations - 1:
                        if verbose:
                            print(f"  ⚠️  Iteration {iteration + 1} failed, refining...")
                            print(f"     Reasoning: {verification_result.reasoning}")

                        # Build feedback from verification failures
                        feedback = f"Goal not achieved. {verification_result.reasoning}"
                        if verification_result.failures:
                            feedback += f"\nFailures: {'; '.join(verification_result.failures)}"
                        if verification_result.recommendations:
                            feedback += f"\nRecommendations: {'; '.join(verification_result.recommendations)}"

                        # Refine code based on verification feedback
                        refinement_result = self.refine_code(
                            original_code=current_code,
                            validation_feedback=feedback,
                            specification=task.specification
                        )

                        current_code = refinement_result.refined_code
                        iterations += 1
                    else:
                        if verbose:
                            print(f"  ❌ Max iterations reached, using best attempt")
            else:
                # Verification disabled - still collect evidence
                evidence = EvidenceCollector.collect_for_code_generation(
                    output_path=task.output_path,
                    test_path=task.test_path,
                    language=task.language
                )
                verification_result = self._verify_goal(goal, evidence, verbose=False)

            # Stage 4: Generate tests if not already done
            if self.config.generate_tests and tests is None:
                if verbose:
                    print(f"[3/3] Generating tests...")

                test_result = self.generate_tests(
                    code=current_code,
                    specification=task.specification,
                    test_framework=self.config.language_config.get("test_framework", "pytest")
                )

                tests = test_result.tests

                # Write tests to disk if enabled and path provided
                if task.test_path and tests:
                    self._write_files(tests, task.test_path, verbose=verbose)

            # Build result with goal-based verification
            result = CodeGenerationResult(
                code=current_code,
                tests=tests,
                documentation=None,  # TODO: Generate docs separately
                goal_achieved=verification_result.is_success if verification_result else False,
                verification_result=verification_result,
                evidence=evidence,
                iterations=iterations,
                metadata={
                    "language": task.language,
                    "mode": self.config.mode,
                    "tier": str(self.tier),
                    "validation_enabled": self.config.validation_enabled,
                    "goal_objective": goal.objective,
                    "required_artifacts": goal.required_artifacts
                }
            )

            # OBSERVABILITY: Emit code generation completed event
            event_store.append(VSMEvent(
                tier="System1_Developer",
                event_type="codegen_completed",
                data={
                    "language": task.language,
                    "goal_achieved": result.goal_achieved,
                    "iterations": iterations,
                    "tests_generated": tests is not None,
                    "correlation_id": correlation_id
                }
            ))

            logger.info(
                f"System 1 (Developer) code generation completed: {'success' if result.goal_achieved else 'failed'}",
                extra={"correlation_id": correlation_id, "goal_achieved": result.goal_achieved}
            )

            if verbose:
                print(f"\n{'=' * 80}")
                print(f"Code generation complete!")
                print(f"  Iterations: {iterations}")
                print(f"  Goal achieved: {'✅ Yes' if result.goal_achieved else '❌ No'}")
                if verification_result:
                    print(f"  Verification score: {verification_result.score:.2f}")
                    print(f"  Artifacts verified: {len(evidence.artifacts_verified)}/{len(goal.required_artifacts)}")
                print(f"  Tests generated: {'✅ Yes' if tests else '❌ No'}")
                print(f"{'=' * 80}\n")

            return result

    def _gather_context(self, task: CodeGenerationTask, verbose: bool = False):
        """
        Gather context from multiple sources using intelligent context preparation.

        Uses ContextPreparationAgent to query:
        1. GraphRAG for past code patterns
        2. ShortTermMemory for recent similar tasks
        3. ObsidianVault for coding standards
        4. Provided task context

        Returns:
            ContextPackage with formatted context, or None if context prep disabled
        """
        if not self.context_prep_agent:
            # Fallback: return task context as simple dict
            return None

        if verbose:
            print(f"  Preparing context for code generation...")

        # Prepare context using intelligent agent
        context_package = self.context_prep_agent.prepare_context(
            user_task=f"Generate {task.language} code: {task.specification}",
            agent_type="developer",
            verbose=False
        )

        if verbose:
            print(f"  ✓ Context prepared (confidence={context_package.confidence:.2f})")
            print(f"    Sources: {', '.join(context_package.sources_used)}")
            print(f"    Token budget: ~{context_package.total_tokens}")

        return context_package

    def _write_files(
        self,
        code: str,
        path: str,
        verbose: bool = False
    ) -> bool:
        """
        Write code to file if file writing is enabled.

        Args:
            code: Code string to write
            path: File path to write to
            verbose: Print status messages

        Returns:
            True if file was written, False otherwise
        """
        if not self.config.enable_file_writing:
            if verbose:
                print(f"  ⚠️  File writing disabled, skipping: {path}")
            return False

        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(code, encoding='utf-8')

            # Verify file was actually written
            if not file_path.exists():
                logger.error(f"Write succeeded but file missing: {path}")
                if verbose:
                    print(f"  ✗ File not found after write: {path}")
                return False

            if verbose:
                file_size = file_path.stat().st_size
                print(f"  ✓ Verified: {path} ({file_size} bytes)")

            return True

        except Exception as e:
            logger.error(f"Failed to write file {path}: {e}")
            if verbose:
                print(f"  ✗ Failed to write {path}: {e}")
            return False

    def _verify_goal(
        self,
        goal: Goal,
        evidence: Evidence,
        verbose: bool = False
    ) -> VerificationResult:
        """
        Verify that the goal was achieved using evidence.

        This replaces the old _validate_code() method with goal-based verification.

        Key Difference:
        - Old: Checked if code STRING parses syntactically
        - New: Checks if GOAL was achieved (files exist, tests pass, etc.)

        Args:
            goal: The goal to verify (e.g., "create Calculator.py at /tmp/calc.py")
            evidence: Evidence collected (files created, tests run, etc.)
            verbose: Print verification details

        Returns:
            VerificationResult with status, score, reasoning, failures, recommendations
        """
        if verbose:
            print(f"  🔍 Verifying goal: {goal.objective}")

        # Use verification module to verify goal
        result = verify_goal(goal, evidence, use_llm=False)  # LLM verification disabled for speed

        if verbose:
            if result.is_success:
                print(f"  ✅ Goal achieved (score: {result.score:.2f})")
            else:
                print(f"  ❌ Goal not achieved (score: {result.score:.2f})")
                print(f"     Reasoning: {result.reasoning}")
                if result.failures:
                    print(f"     Failures: {', '.join(result.failures)}")

        return result


# ============================================================================
# Module entry point for testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DeveloperAgent - Quick Test")
    print("=" * 80)
    print()

    # Test configuration
    config = PresetDeveloperConfigs.quick_python()
    agent = DeveloperAgent(config=config)

    # Test task
    task = CodeGenerationTask(
        specification="""
        Create a simple Calculator class with:
        - add(a, b) -> float
        - subtract(a, b) -> float
        - multiply(a, b) -> float
        - divide(a, b) -> float (handle division by zero)

        Include docstrings and type hints.
        """,
        language="python",
        context="""
        Example pattern:
        ```python
        class Example:
            '''Example class with docstrings.'''

            def method(self, x: int) -> int:
                '''Method with type hints.'''
                return x * 2
        ```
        """
    )

    # Generate code
    print("Generating Calculator class...")
    result = agent(task, verbose=True)

    print("Generated Code:")
    print("=" * 80)
    print(result.code)
    print("=" * 80)
    print()

    if result.tests:
        print("Generated Tests:")
        print("=" * 80)
        print(result.tests)
        print("=" * 80)
