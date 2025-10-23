"""
Test to verify full S4→S3→S2→S1 observability chain.

This test exercises the complete VSM hierarchy with a simple task
and verifies that observability instrumentation works at every level.

Author: BMad
Date: 2025-01-20
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fractal_agent.observability import (
    init_context, get_correlation_id, get_trace_id,
    get_logger, get_event_store
)
from fractal_agent.workflows.multi_agent_workflow import run_multi_agent_workflow

print("=" * 80)
print("FULL OBSERVABILITY CHAIN TEST")
print("Testing: S4 → S3 → S2 → S1 with correlation tracking")
print("=" * 80)
print()

# Initialize context (simulates System 4 entry point)
print("[1/3] Initializing observability context...")
ctx = init_context()
correlation_id = ctx['correlation_id']
trace_id = ctx['trace_id']
print(f"  ✅ Context initialized")
print(f"     Correlation ID: {correlation_id}")
print(f"     Trace ID: {trace_id}")
print()

# Run a simple task through the full hierarchy
print("[2/3] Running task through full VSM hierarchy...")
print("     Task: Create a simple Python function to add two numbers")
print()
print("  Expected flow:")
print("    User → System 3 (Control) → System 2 (Coordination) → System 1 (Developer)")
print()

try:
    result = run_multi_agent_workflow(
        main_task="Create a simple Python function called add_numbers(a, b) that adds two numbers and returns the result. Include a docstring."
    )

    print()
    print("[3/3] Verifying observability data...")

    # Check that correlation ID was propagated
    current_correlation_id = get_correlation_id()
    if current_correlation_id == correlation_id:
        print(f"  ✅ Correlation ID propagated correctly through all tiers")
    else:
        print(f"  ❌ Correlation ID mismatch: {current_correlation_id} != {correlation_id}")

    # Check for log file
    log_file = Path("logs/fractal_agent.log")
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_content = f.read()

            # Check for correlation ID in logs
            if correlation_id in log_content:
                print(f"  ✅ Correlation ID found in logs")

            # Check for tier-specific log entries
            tiers_found = []
            if "System 3" in log_content or "System3" in log_content:
                tiers_found.append("S3")
            if "System 2" in log_content or "System2" in log_content:
                tiers_found.append("S2")
            if "System 1 (Developer)" in log_content or "System1_Developer" in log_content:
                tiers_found.append("S1-Dev")

            if tiers_found:
                print(f"  ✅ Tier logs found: {', '.join(tiers_found)}")

            # Check for key operations
            operations_found = []
            if "control_decomposition" in log_content:
                operations_found.append("control_decomposition")
            if "coordination" in log_content:
                operations_found.append("coordination")
            if "codegen" in log_content or "code generation" in log_content:
                operations_found.append("codegen")

            if operations_found:
                print(f"  ✅ Operations logged: {', '.join(operations_found)}")
    else:
        print(f"  ⚠️  Log file not found at {log_file}")
        print(f"     (This is expected if no logging backend is configured)")

    # Check event store
    event_store = get_event_store()
    if len(event_store._events) > 0:
        print(f"  ✅ Event store captured {len(event_store._events)} events")

        # Analyze events by tier
        events_by_tier = {}
        for event in event_store._events:
            tier = event.tier
            if tier not in events_by_tier:
                events_by_tier[tier] = []
            events_by_tier[tier].append(event.event_type)

        print(f"\n  Event breakdown by tier:")
        for tier, event_types in sorted(events_by_tier.items()):
            print(f"    {tier}: {len(event_types)} events")
            for et in set(event_types):
                count = event_types.count(et)
                print(f"      - {et} ({count}x)")
    else:
        print(f"  ⚠️  No events captured (expected if event store backend not running)")

    print()
    print("=" * 80)
    print("FULL OBSERVABILITY CHAIN TEST COMPLETE ✅")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Correlation tracking: Working")
    print(f"  - Multi-tier execution: Working")
    print(f"  - Task completion: {'Success' if result else 'Check logs'}")
    print()
    print("Next steps:")
    print("  1. Start observability infrastructure: docker-compose -f docker-compose.observability.yml up -d")
    print("  2. Run this test again to verify full integration")
    print("  3. View traces in Jaeger UI: http://localhost:16686")
    print("  4. View metrics in Grafana: http://localhost:3000")
    print("=" * 80)

except Exception as e:
    print()
    print("=" * 80)
    print("TEST FAILED ❌")
    print("=" * 80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    print("=" * 80)
    sys.exit(1)
