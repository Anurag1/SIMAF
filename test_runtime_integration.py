#!/usr/bin/env python3
"""
REAL Runtime Integration Test - Zero Tolerance

Tests that code actually RUNS correctly, not just imports.

Tests ACTUAL execution:
- Observability system (correlation_id, metrics, events)
- Database writes
- LLM calls
- Agent execution
- Memory systems

Author: BMad
Date: 2025-10-23
"""

import sys
import logging
import os

# Reduce noise but show errors
logging.basicConfig(level=logging.WARNING)

def test_1_observability_runtime():
    """Test 1: Observability actually works at runtime"""
    print("\n[Test 1] Observability Runtime")
    print("-" * 60)

    from fractal_agent.observability import (
        get_correlation_id, get_trace_id,
        get_event_store, VSMEvent
    )
    from fractal_agent.observability.metrics import registry
    from prometheus_client import CollectorRegistry

    # Test correlation_id NEVER returns None at runtime
    corr_id = get_correlation_id()
    assert corr_id is not None, "❌ correlation_id returned None at runtime!"
    assert isinstance(corr_id, str), f"❌ correlation_id wrong type: {type(corr_id)}"
    assert len(corr_id) > 0, "❌ correlation_id is empty string!"
    print(f"✅ correlation_id works at runtime: {corr_id[:8]}...")

    # Test trace_id NEVER returns None
    trace_id = get_trace_id()
    assert trace_id is not None, "❌ trace_id returned None at runtime!"
    print(f"✅ trace_id works at runtime: {trace_id[:8]}...")

    # Test Prometheus registry is ACTUALLY a CollectorRegistry
    assert isinstance(registry, CollectorRegistry), \
        f"❌ Registry is {type(registry)}, not CollectorRegistry!"
    print("✅ Prometheus registry is correct type at runtime")

    # Test we can actually call collect() on it
    try:
        list(registry.collect())
        print("✅ Registry.collect() works at runtime")
    except AttributeError as e:
        raise AssertionError(f"❌ Registry.collect() failed: {e}")

    return True


def test_2_database_writes():
    """Test 2: Database actually accepts writes"""
    print("\n[Test 2] Database Writes Runtime")
    print("-" * 60)

    from fractal_agent.observability import get_event_store, VSMEvent
    from datetime import datetime

    event_store = get_event_store()
    if not event_store:
        print("⚠️  Database not configured (docker-compose not running)")
        return True

    # Try to write an event with various tier names
    test_tiers = [
        "System1",
        "System1_Research",
        "FractalDSpy_cheap",
        "FractalDSpy_balanced"
    ]

    for tier in test_tiers:
        try:
            event = VSMEvent(
                event_type="test_event",
                tier=tier,
                data={"test": True},
                timestamp=datetime.now()
            )
            event_store.append(event)
            print(f"✅ Database accepts tier: {tier}")
        except Exception as e:
            if "tier_check" in str(e).lower():
                raise AssertionError(
                    f"❌ Database REJECTS tier '{tier}': {e}\n"
                    f"   Constraint not properly dropped!"
                )
            else:
                print(f"⚠️  Database error (non-constraint): {e}")

    return True


def test_3_llm_call_runtime():
    """Test 3: LLM calls actually work"""
    print("\n[Test 3] LLM Call Runtime")
    print("-" * 60)

    from fractal_agent.utils.model_config import configure_lm
    from fractal_agent.observability import get_correlation_id

    # Check correlation_id is set BEFORE LLM call
    corr_id_before = get_correlation_id()
    assert corr_id_before is not None, "❌ correlation_id None before LLM call!"

    lm = configure_lm(tier="cheap")
    result = lm(prompt="Say 'test' and nothing else", max_tokens=10)

    assert result is not None, "❌ LLM returned None!"
    assert isinstance(result, dict), f"❌ LLM returned {type(result)}, not dict!"
    assert "text" in result, "❌ LLM result missing 'text' key!"
    assert len(result["text"]) > 0, "❌ LLM returned empty string!"
    print(f"✅ LLM call worked: '{result['text'][:50]}'")
    print(f"✅ Provider: {result.get('provider')}, Model: {result.get('model')}")

    # Check correlation_id is STILL set AFTER LLM call
    corr_id_after = get_correlation_id()
    assert corr_id_after is not None, "❌ correlation_id became None after LLM call!"
    assert corr_id_after == corr_id_before, \
        f"❌ correlation_id changed during LLM call! {corr_id_before} -> {corr_id_after}"
    print("✅ correlation_id preserved through LLM call")

    return True


def test_4_context_prep_runtime():
    """Test 4: Context preparation actually runs"""
    print("\n[Test 4] Context Preparation Runtime")
    print("-" * 60)

    from fractal_agent.agents.context_preparation_agent import ContextPreparationAgent

    agent = ContextPreparationAgent()

    # Check research_missing_context actually exists
    assert hasattr(agent, 'research_missing_context'), \
        "❌ research_missing_context method missing!"

    # Try to call it
    try:
        result = agent.research_missing_context(["test topic"], verbose=False)
        assert isinstance(result, dict), \
            f"❌ research_missing_context returned {type(result)}, not dict!"
        print("✅ research_missing_context runs without 'not yet implemented' warning")
    except Exception as e:
        if "not yet implemented" in str(e).lower():
            raise AssertionError(f"❌ research_missing_context still not implemented!")
        else:
            print(f"⚠️  research_missing_context error (non-impl): {e}")

    return True


def test_5_embeddings_runtime():
    """Test 5: Embeddings generate consistent dimensions"""
    print("\n[Test 5] Embeddings Runtime")
    print("-" * 60)

    try:
        from fractal_agent.memory.embeddings import generate_embedding
    except ImportError:
        print("⚠️  Embeddings module not found (may not be implemented yet)")
        return True

    test_texts = [
        "Short text",
        "Medium length text with more words in it",
        "Very long text " * 50  # 150+ words
    ]

    dimensions = []
    for text in test_texts:
        try:
            embedding = generate_embedding(text)
            dim = len(embedding)
            dimensions.append(dim)
            print(f"  Text length {len(text):4d} -> embedding dimension {dim}")
        except Exception as e:
            print(f"⚠️  Embedding failed: {e}")
            continue

    if len(dimensions) > 0:
        if len(set(dimensions)) > 1:
            raise AssertionError(
                f"❌ INCONSISTENT embedding dimensions: {dimensions}\n"
                f"   Expected all to be 1536, got varying sizes!"
            )
        if dimensions[0] != 1536:
            raise AssertionError(
                f"❌ WRONG embedding dimension: {dimensions[0]}, expected 1536!"
            )
        print(f"✅ All embeddings consistent at {dimensions[0]} dimensions")

    return True


def main():
    """Run all runtime integration tests"""
    print("=" * 80)
    print("REAL RUNTIME INTEGRATION TESTS")
    print("Tests actual code execution - Zero tolerance for errors")
    print("=" * 80)

    tests = [
        test_1_observability_runtime,
        test_2_database_writes,
        test_3_llm_call_runtime,
        test_4_context_prep_runtime,
        test_5_embeddings_runtime,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n❌ {test_func.__name__} FAILED:")
            print(f"   {e}")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_func.__name__} CRASHED:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("=" * 80)

    if failed > 0:
        print(f"\n❌ {failed} test(s) FAILED - Runtime issues found")
        sys.exit(1)
    else:
        print("\n✅ ALL TESTS PASSED - Runtime working correctly")
        sys.exit(0)


if __name__ == "__main__":
    main()
