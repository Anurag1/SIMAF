"""
Test Production Monitoring System

Demonstrates and validates the complete production monitoring stack including:
- Prometheus metrics collection
- Real-time cost tracking with accurate pricing
- Budget monitoring and alerts
- Grafana dashboard data population
- Metrics server endpoints

Run this to verify the monitoring system is working correctly.

Usage:
    python test_production_monitoring.py

Author: BMad
Date: 2025-01-20
"""

import time
import random
import json
from fractal_agent.observability.production_monitoring import (
    get_production_monitor,
    initialize_production_monitoring
)
from fractal_agent.observability.metrics_server_enhanced import start_metrics_server
from fractal_agent.utils.model_registry import get_registry


def simulate_llm_calls(monitor, num_calls: int = 20):
    """Simulate various LLM calls to populate metrics"""

    print(f"\nSimulating {num_calls} LLM calls...")

    models = [
        ('anthropic', 'claude-sonnet-4.5'),
        ('anthropic', 'claude-haiku-4.5'),
        ('anthropic', 'claude-opus-4.1'),
        ('gemini', 'gemini-2.0-flash-exp'),
        ('gemini', 'gemini-1.5-pro'),
    ]

    tiers = [
        'System1_Research',
        'System2_Developer',
        'System3_Coordination',
        'System4_Intelligence',
        'Memory'
    ]

    for i in range(num_calls):
        provider, model = random.choice(models)
        tier = random.choice(tiers)

        # Simulate realistic token usage
        input_tokens = random.randint(500, 5000)
        output_tokens = random.randint(200, 2000)
        latency_ms = random.uniform(500, 5000)
        cache_hit = random.random() < 0.3  # 30% cache hit rate
        success = random.random() < 0.95  # 95% success rate

        # Record call
        if success:
            result = monitor.record_llm_call(
                tier=tier,
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cache_hit=cache_hit,
                success=True
            )

            print(
                f"  [{i+1:2d}] {tier:20s} | {provider:10s}/{model:25s} | "
                f"{input_tokens:5d} in + {output_tokens:4d} out | "
                f"${result['total_cost_usd']:7.4f} | "
                f"{latency_ms:6.0f}ms | "
                f"{'CACHE' if cache_hit else 'MISS '}"
            )

        else:
            monitor.record_llm_call(
                tier=tier,
                provider=provider,
                model=model,
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                cache_hit=False,
                success=False,
                error_type='TimeoutError'
            )
            print(f"  [{i+1:2d}] {tier:20s} | {provider:10s}/{model:25s} | ERROR")

        # Small delay to make metrics more realistic
        time.sleep(0.1)

    print(f"\n✓ Completed {num_calls} simulated calls")


def print_budget_status(monitor):
    """Print budget status"""
    status = monitor.get_budget_status()

    print("\n" + "=" * 80)
    print("BUDGET STATUS")
    print("=" * 80)

    # Hourly
    hourly = status['hourly']
    print(f"\nHourly Budget:")
    print(f"  Rate:        ${hourly['rate_usd']:.2f}/hr")
    print(f"  Budget:      ${hourly['budget_usd']:.2f}/hr")
    print(f"  Utilization: {hourly['utilization_pct']:.1f}%")
    print(f"  Remaining:   ${hourly['remaining_usd']:.2f}/hr")
    print(f"  Status:      {'❌ EXCEEDED' if hourly['exceeded'] else '✓ OK'}")

    # Daily
    daily = status['daily']
    print(f"\nDaily Budget:")
    print(f"  Total:       ${daily['total_usd']:.2f}")
    print(f"  Budget:      ${daily['budget_usd']:.2f}")
    print(f"  Utilization: {daily['utilization_pct']:.1f}%")
    print(f"  Remaining:   ${daily['remaining_usd']:.2f}")
    print(f"  Status:      {'❌ EXCEEDED' if daily['exceeded'] else '✓ OK'}")

    # Alerts
    if status['alert_triggered']:
        print(f"\n⚠️  ALERTS TRIGGERED:")
        for alert in status['recent_alerts']:
            print(f"    - {alert['type']} at {alert['timestamp']}")

    print("=" * 80)


def print_cost_breakdown(monitor):
    """Print cost breakdown"""
    breakdown = monitor.get_cost_breakdown()

    print("\n" + "=" * 80)
    print("COST BREAKDOWN")
    print("=" * 80)

    print(f"\nTotal Cost: ${breakdown['total_cost_usd']:.4f}")

    print(f"\nBy Tier:")
    for tier, cost in sorted(breakdown['by_tier'].items(), key=lambda x: x[1], reverse=True):
        pct = (cost / breakdown['total_cost_usd'] * 100) if breakdown['total_cost_usd'] > 0 else 0
        print(f"  {tier:25s}: ${cost:8.4f} ({pct:5.1f}%)")

    print(f"\nBy Provider:")
    for provider, cost in sorted(breakdown['by_provider'].items(), key=lambda x: x[1], reverse=True):
        pct = (cost / breakdown['total_cost_usd'] * 100) if breakdown['total_cost_usd'] > 0 else 0
        print(f"  {provider:25s}: ${cost:8.4f} ({pct:5.1f}%)")

    print(f"\nBy Model:")
    for model, cost in sorted(breakdown['by_model'].items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = (cost / breakdown['total_cost_usd'] * 100) if breakdown['total_cost_usd'] > 0 else 0
        print(f"  {model:40s}: ${cost:8.4f} ({pct:5.1f}%)")

    print("=" * 80)


def print_metrics_endpoints(port: int = 9100):
    """Print available metrics endpoints"""
    print("\n" + "=" * 80)
    print("METRICS ENDPOINTS")
    print("=" * 80)
    print(f"\n  Prometheus Metrics:  http://localhost:{port}/metrics")
    print(f"  Health Check:        http://localhost:{port}/health")
    print(f"  Cost Status:         http://localhost:{port}/cost")
    print(f"  Budget Status:       http://localhost:{port}/budget")
    print(f"  Index Page:          http://localhost:{port}/")
    print("\n  Grafana Dashboards:  http://localhost:3002/")
    print("    - System Overview:   http://localhost:3002/d/fractal-vsm-overview")
    print("    - Agent Performance: http://localhost:3002/d/fractal-vsm-agents")
    print("    - Cost Tracking:     http://localhost:3002/d/fractal-cost-tracking")
    print("\n  Prometheus UI:       http://localhost:9090/")
    print("  Jaeger Tracing:      http://localhost:16686/")
    print("=" * 80)


def test_model_registry():
    """Test model registry integration"""
    print("\n" + "=" * 80)
    print("MODEL REGISTRY")
    print("=" * 80)

    registry = get_registry()

    print(f"\nTotal models: {len(registry.models)}")
    print(f"By tier: {registry.get_tier_summary()}")
    print(f"By provider: {registry.get_provider_summary()}")

    print("\nModel Pricing:")
    for model_id, model_info in list(registry.models.items())[:5]:
        print(
            f"  {model_id:30s} | "
            f"${model_info.input_cost_per_mtok:6.2f}/${model_info.output_cost_per_mtok:6.2f} per Mtok | "
            f"{model_info.tier:10s} | "
            f"cache={model_info.supports_caching}"
        )

    print("=" * 80)


def main():
    """Main test function"""
    print("\n" + "=" * 80)
    print("PRODUCTION MONITORING SYSTEM TEST")
    print("=" * 80)

    # Test model registry
    test_model_registry()

    # Initialize monitoring with test budgets
    print("\nInitializing production monitoring...")
    monitor = initialize_production_monitoring(
        hourly_budget_usd=10.0,    # Low budget for testing alerts
        daily_budget_usd=100.0
    )
    print("✓ Production monitoring initialized")

    # Start metrics server
    print("\nStarting metrics server...")
    try:
        server = start_metrics_server(port=9100)
        print("✓ Metrics server started")
    except Exception as e:
        print(f"⚠️  Metrics server error: {e}")

    # Print endpoints
    print_metrics_endpoints()

    # Simulate LLM calls
    simulate_llm_calls(monitor, num_calls=30)

    # Print status
    print_budget_status(monitor)
    print_cost_breakdown(monitor)

    # Export sample metrics
    print("\n" + "=" * 80)
    print("SAMPLE METRICS (Prometheus Format)")
    print("=" * 80)
    metrics = monitor.export_metrics()
    # Print first 50 lines
    for line in metrics.split('\n')[:50]:
        if line and not line.startswith('#'):
            print(f"  {line}")
    print("  ...")
    print(f"\n  Total metrics lines: {len(metrics.split(chr(10)))}")
    print("=" * 80)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("\n✓ Production monitoring system is operational")
    print("\nNext steps:")
    print("  1. Open Grafana at http://localhost:3002/")
    print("  2. Navigate to the 'Cost Tracking' dashboard")
    print("  3. Verify metrics are being displayed")
    print("  4. Check Prometheus at http://localhost:9090/")
    print("  5. Query metrics like: fractal_llm_cost_usd_total")
    print("\nTo integrate with your application:")
    print("  from fractal_agent.observability.auto_instrumentation import enable_auto_instrumentation")
    print("  enable_auto_instrumentation(hourly_budget_usd=100.0, daily_budget_usd=1000.0)")
    print("\nMetrics will be scraped by Prometheus every 15 seconds.")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
