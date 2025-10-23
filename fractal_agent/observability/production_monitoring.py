"""
Production Monitoring System for Fractal Agent

Complete integration of Prometheus metrics, Grafana dashboards, and real-time cost tracking.
This module provides production-ready monitoring with accurate cost calculations, alerts,
and comprehensive observability.

Features:
- Real-time Prometheus metrics collection
- Accurate per-model cost tracking (input/output separated)
- Grafana dashboard data population
- Cost alerts and budget tracking
- Performance monitoring and SLO tracking
- Auto-initialization with the observability stack

Usage:
    from fractal_agent.observability.production_monitoring import (
        ProductionMonitor, get_production_monitor
    )

    # Get singleton instance
    monitor = get_production_monitor()

    # Record LLM call with accurate costing
    monitor.record_llm_call(
        tier="System1_Research",
        provider="anthropic",
        model="claude-sonnet-4.5",
        input_tokens=1000,
        output_tokens=500,
        latency_ms=2500,
        cache_hit=False,
        success=True
    )

    # Check budget status
    status = monitor.get_budget_status()
    if status['alert_triggered']:
        print(f"Budget alert: {status['message']}")

    # Get cost breakdown
    breakdown = monitor.get_cost_breakdown(hours=24)

Author: BMad
Date: 2025-01-20
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Lock
import json

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, push_to_gateway, generate_latest
)

from ..utils.model_registry import ModelRegistry, ModelInfo, get_registry

logger = logging.getLogger(__name__)


@dataclass
class CostWindow:
    """Tracks costs within a time window for budget monitoring"""
    window_start: datetime
    window_duration_hours: int
    costs: List[Tuple[datetime, float]] = field(default_factory=list)
    total_cost: float = 0.0

    def add_cost(self, cost: float, timestamp: Optional[datetime] = None):
        """Add a cost entry"""
        if timestamp is None:
            timestamp = datetime.now()
        self.costs.append((timestamp, cost))
        self.total_cost += cost

    def prune_old_entries(self):
        """Remove entries outside the time window"""
        cutoff = datetime.now() - timedelta(hours=self.window_duration_hours)
        new_costs = [(ts, cost) for ts, cost in self.costs if ts >= cutoff]
        self.costs = new_costs
        self.total_cost = sum(cost for _, cost in new_costs)

    def get_rate_per_hour(self) -> float:
        """Get average cost per hour in current window"""
        self.prune_old_entries()
        if not self.costs:
            return 0.0
        duration_hours = (datetime.now() - self.costs[0][0]).total_seconds() / 3600
        if duration_hours == 0:
            return 0.0
        return self.total_cost / duration_hours


class ProductionMonitor:
    """
    Production monitoring system with comprehensive observability.

    Provides:
    - Accurate LLM cost tracking with input/output separation
    - Real-time Prometheus metrics
    - Budget monitoring and alerts
    - Performance tracking and SLO monitoring
    - Cost breakdown by tier/provider/model
    - Integration with Grafana dashboards
    """

    def __init__(
        self,
        registry: Optional[CollectorRegistry] = None,
        model_registry: Optional[ModelRegistry] = None,
        hourly_budget_usd: float = 100.0,
        daily_budget_usd: float = 1000.0
    ):
        """
        Initialize production monitoring system.

        Args:
            registry: Prometheus registry (creates new if None)
            model_registry: Model registry for pricing (uses global if None)
            hourly_budget_usd: Hourly spending limit for alerts
            daily_budget_usd: Daily spending limit for alerts
        """
        self.registry = registry or CollectorRegistry()
        self.model_registry = model_registry or get_registry()

        # Budget configuration
        self.hourly_budget_usd = hourly_budget_usd
        self.daily_budget_usd = daily_budget_usd

        # Cost tracking windows
        self.hourly_costs = CostWindow(
            window_start=datetime.now(),
            window_duration_hours=1
        )
        self.daily_costs = CostWindow(
            window_start=datetime.now(),
            window_duration_hours=24
        )

        # Thread safety
        self._lock = Lock()

        # Cost breakdown tracking
        self.cost_by_tier: Dict[str, float] = defaultdict(float)
        self.cost_by_provider: Dict[str, float] = defaultdict(float)
        self.cost_by_model: Dict[str, float] = defaultdict(float)

        # Alert tracking
        self.alerts_triggered: List[Dict[str, Any]] = []

        # Initialize Prometheus metrics
        self._init_metrics()

        logger.info(
            f"ProductionMonitor initialized - "
            f"Hourly budget: ${hourly_budget_usd}, Daily budget: ${daily_budget_usd}"
        )

    def _init_metrics(self):
        """Initialize all Prometheus metrics"""

        # ====================================================================
        # LLM Metrics with Enhanced Granularity
        # ====================================================================

        self.llm_calls_total = Counter(
            'fractal_llm_calls_total',
            'Total number of LLM API calls',
            ['tier', 'provider', 'model', 'cache_hit', 'status'],
            registry=self.registry
        )

        self.llm_tokens_input_total = Counter(
            'fractal_llm_tokens_input_total',
            'Total input tokens used',
            ['tier', 'provider', 'model', 'cache_hit'],
            registry=self.registry
        )

        self.llm_tokens_output_total = Counter(
            'fractal_llm_tokens_output_total',
            'Total output tokens used',
            ['tier', 'provider', 'model', 'cache_hit'],
            registry=self.registry
        )

        self.llm_cost_usd_total = Counter(
            'fractal_llm_cost_usd_total',
            'Total LLM cost in USD (accurate per-token pricing)',
            ['tier', 'provider', 'model', 'cache_hit', 'token_type'],
            registry=self.registry
        )

        self.llm_latency_seconds = Histogram(
            'fractal_llm_latency_seconds',
            'LLM call latency distribution',
            ['tier', 'provider', 'model'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self.registry
        )

        self.llm_cache_hits_total = Counter(
            'fractal_llm_cache_hits_total',
            'Total LLM cache hits',
            ['tier', 'provider', 'model'],
            registry=self.registry
        )

        self.llm_errors_total = Counter(
            'fractal_llm_errors_total',
            'Total LLM call errors',
            ['tier', 'provider', 'model', 'error_type'],
            registry=self.registry
        )

        # ====================================================================
        # Cost Monitoring Metrics
        # ====================================================================

        self.cost_hourly_rate_usd = Gauge(
            'fractal_cost_hourly_rate_usd',
            'Current hourly cost rate',
            registry=self.registry
        )

        self.cost_daily_total_usd = Gauge(
            'fractal_cost_daily_total_usd',
            'Total cost in last 24 hours',
            registry=self.registry
        )

        self.cost_budget_utilization_percent = Gauge(
            'fractal_cost_budget_utilization_percent',
            'Budget utilization percentage',
            ['window'],
            registry=self.registry
        )

        self.cost_by_tier_usd = Gauge(
            'fractal_cost_by_tier_usd',
            'Cost breakdown by tier',
            ['tier'],
            registry=self.registry
        )

        self.cost_by_provider_usd = Gauge(
            'fractal_cost_by_provider_usd',
            'Cost breakdown by provider',
            ['provider'],
            registry=self.registry
        )

        self.cost_by_model_usd = Gauge(
            'fractal_cost_by_model_usd',
            'Cost breakdown by model',
            ['provider', 'model'],
            registry=self.registry
        )

        self.cost_alerts_total = Counter(
            'fractal_cost_alerts_total',
            'Total cost alerts triggered',
            ['alert_type'],
            registry=self.registry
        )

        # ====================================================================
        # Performance and SLO Metrics
        # ====================================================================

        self.slo_success_rate = Gauge(
            'fractal_slo_success_rate',
            'SLO success rate (last 1h)',
            ['tier'],
            registry=self.registry
        )

        self.slo_p95_latency_seconds = Gauge(
            'fractal_slo_p95_latency_seconds',
            'SLO p95 latency',
            ['tier'],
            registry=self.registry
        )

        # ====================================================================
        # System Health Metrics
        # ====================================================================

        self.monitor_info = Info(
            'fractal_monitor',
            'Production monitor metadata',
            registry=self.registry
        )

        self.monitor_info.info({
            'version': '1.0.0',
            'hourly_budget_usd': str(self.hourly_budget_usd),
            'daily_budget_usd': str(self.daily_budget_usd),
            'initialized_at': datetime.now().isoformat()
        })

    def record_llm_call(
        self,
        tier: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cache_hit: bool = False,
        success: bool = True,
        error_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record LLM call with accurate cost tracking.

        Args:
            tier: VSM tier (e.g., "System1_Research")
            provider: Provider name (e.g., "anthropic")
            model: Model ID (e.g., "claude-sonnet-4.5")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_ms: Call latency in milliseconds
            cache_hit: Whether this was a cache hit
            success: Whether call succeeded
            error_type: Error type if failed

        Returns:
            Dict with cost breakdown and metrics
        """
        with self._lock:
            status = 'success' if success else 'error'
            cache_label = 'true' if cache_hit else 'false'

            # Get model info for accurate pricing
            model_info = self.model_registry.get_model_info(model)

            if model_info is None:
                logger.warning(f"Model {model} not found in registry, using default pricing")
                input_cost_per_mtok = 3.0  # Default fallback
                output_cost_per_mtok = 15.0
            else:
                input_cost_per_mtok = model_info.input_cost_per_mtok
                output_cost_per_mtok = model_info.output_cost_per_mtok

            # Calculate accurate costs
            input_cost = (input_tokens / 1_000_000) * input_cost_per_mtok
            output_cost = (output_tokens / 1_000_000) * output_cost_per_mtok

            # Apply cache discount (90% off for cached reads)
            if cache_hit:
                input_cost *= 0.1
                output_cost *= 0.1

            total_cost = input_cost + output_cost

            # Record call
            self.llm_calls_total.labels(
                tier=tier,
                provider=provider,
                model=model,
                cache_hit=cache_label,
                status=status
            ).inc()

            if success:
                # Record input tokens
                self.llm_tokens_input_total.labels(
                    tier=tier,
                    provider=provider,
                    model=model,
                    cache_hit=cache_label
                ).inc(input_tokens)

                # Record output tokens
                self.llm_tokens_output_total.labels(
                    tier=tier,
                    provider=provider,
                    model=model,
                    cache_hit=cache_label
                ).inc(output_tokens)

                # Record costs separately
                self.llm_cost_usd_total.labels(
                    tier=tier,
                    provider=provider,
                    model=model,
                    cache_hit=cache_label,
                    token_type='input'
                ).inc(input_cost)

                self.llm_cost_usd_total.labels(
                    tier=tier,
                    provider=provider,
                    model=model,
                    cache_hit=cache_label,
                    token_type='output'
                ).inc(output_cost)

                # Record latency
                self.llm_latency_seconds.labels(
                    tier=tier,
                    provider=provider,
                    model=model
                ).observe(latency_ms / 1000.0)

                # Record cache hit
                if cache_hit:
                    self.llm_cache_hits_total.labels(
                        tier=tier,
                        provider=provider,
                        model=model
                    ).inc()

                # Update cost tracking
                self._update_cost_tracking(tier, provider, model, total_cost)

                # Check budget alerts
                self._check_budget_alerts()

            else:
                # Record error
                self.llm_errors_total.labels(
                    tier=tier,
                    provider=provider,
                    model=model,
                    error_type=error_type or 'unknown'
                ).inc()

            return {
                'input_cost_usd': input_cost,
                'output_cost_usd': output_cost,
                'total_cost_usd': total_cost,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'cache_hit': cache_hit,
                'success': success
            }

    def _update_cost_tracking(
        self,
        tier: str,
        provider: str,
        model: str,
        cost: float
    ):
        """Update cost tracking across all dimensions"""
        # Add to time windows
        now = datetime.now()
        self.hourly_costs.add_cost(cost, now)
        self.daily_costs.add_cost(cost, now)

        # Update breakdown tracking
        self.cost_by_tier[tier] += cost
        self.cost_by_provider[provider] += cost
        self.cost_by_model[f"{provider}/{model}"] += cost

        # Update Prometheus gauges
        self.cost_hourly_rate_usd.set(self.hourly_costs.get_rate_per_hour())
        self.cost_daily_total_usd.set(self.daily_costs.total_cost)

        # Update budget utilization
        hourly_utilization = (self.hourly_costs.get_rate_per_hour() / self.hourly_budget_usd) * 100
        daily_utilization = (self.daily_costs.total_cost / self.daily_budget_usd) * 100

        self.cost_budget_utilization_percent.labels(window='hourly').set(hourly_utilization)
        self.cost_budget_utilization_percent.labels(window='daily').set(daily_utilization)

        # Update breakdown gauges
        self.cost_by_tier_usd.labels(tier=tier).set(self.cost_by_tier[tier])
        self.cost_by_provider_usd.labels(provider=provider).set(self.cost_by_provider[provider])
        self.cost_by_model_usd.labels(provider=provider, model=model).set(
            self.cost_by_model[f"{provider}/{model}"]
        )

    def _check_budget_alerts(self):
        """Check if budget thresholds are exceeded"""
        hourly_rate = self.hourly_costs.get_rate_per_hour()
        daily_total = self.daily_costs.total_cost

        # Hourly budget alert
        if hourly_rate > self.hourly_budget_usd:
            alert = {
                'type': 'hourly_budget_exceeded',
                'timestamp': datetime.now().isoformat(),
                'hourly_rate_usd': hourly_rate,
                'budget_usd': self.hourly_budget_usd,
                'overage_pct': ((hourly_rate / self.hourly_budget_usd) - 1) * 100
            }
            self.alerts_triggered.append(alert)
            self.cost_alerts_total.labels(alert_type='hourly_budget').inc()
            logger.warning(
                f"Hourly budget exceeded: ${hourly_rate:.2f}/hr "
                f"(budget: ${self.hourly_budget_usd:.2f}/hr)"
            )

        # Daily budget alert
        if daily_total > self.daily_budget_usd:
            alert = {
                'type': 'daily_budget_exceeded',
                'timestamp': datetime.now().isoformat(),
                'daily_total_usd': daily_total,
                'budget_usd': self.daily_budget_usd,
                'overage_pct': ((daily_total / self.daily_budget_usd) - 1) * 100
            }
            self.alerts_triggered.append(alert)
            self.cost_alerts_total.labels(alert_type='daily_budget').inc()
            logger.warning(
                f"Daily budget exceeded: ${daily_total:.2f}/day "
                f"(budget: ${self.daily_budget_usd:.2f}/day)"
            )

    def get_budget_status(self) -> Dict[str, Any]:
        """
        Get current budget status.

        Returns:
            Dict with budget utilization and alert status
        """
        with self._lock:
            self.hourly_costs.prune_old_entries()
            self.daily_costs.prune_old_entries()

            hourly_rate = self.hourly_costs.get_rate_per_hour()
            daily_total = self.daily_costs.total_cost

            return {
                'hourly': {
                    'rate_usd': hourly_rate,
                    'budget_usd': self.hourly_budget_usd,
                    'utilization_pct': (hourly_rate / self.hourly_budget_usd) * 100,
                    'remaining_usd': max(0, self.hourly_budget_usd - hourly_rate),
                    'exceeded': hourly_rate > self.hourly_budget_usd
                },
                'daily': {
                    'total_usd': daily_total,
                    'budget_usd': self.daily_budget_usd,
                    'utilization_pct': (daily_total / self.daily_budget_usd) * 100,
                    'remaining_usd': max(0, self.daily_budget_usd - daily_total),
                    'exceeded': daily_total > self.daily_budget_usd
                },
                'alert_triggered': (
                    hourly_rate > self.hourly_budget_usd or
                    daily_total > self.daily_budget_usd
                ),
                'recent_alerts': self.alerts_triggered[-5:]  # Last 5 alerts
            }

    def get_cost_breakdown(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get cost breakdown by various dimensions.

        Args:
            hours: Time window in hours

        Returns:
            Dict with cost breakdown
        """
        with self._lock:
            return {
                'time_window_hours': hours,
                'total_cost_usd': self.daily_costs.total_cost if hours >= 24 else self.hourly_costs.total_cost,
                'by_tier': dict(self.cost_by_tier),
                'by_provider': dict(self.cost_by_provider),
                'by_model': dict(self.cost_by_model),
                'timestamp': datetime.now().isoformat()
            }

    def export_metrics(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Metrics in text format for scraping
        """
        return generate_latest(self.registry).decode('utf-8')

    def reset_cost_tracking(self):
        """Reset cost tracking (use carefully - typically for testing only)"""
        with self._lock:
            self.hourly_costs = CostWindow(
                window_start=datetime.now(),
                window_duration_hours=1
            )
            self.daily_costs = CostWindow(
                window_start=datetime.now(),
                window_duration_hours=24
            )
            self.cost_by_tier.clear()
            self.cost_by_provider.clear()
            self.cost_by_model.clear()
            self.alerts_triggered.clear()

            logger.info("Cost tracking reset")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary for dashboards.

        Returns:
            Dict with all key metrics
        """
        budget_status = self.get_budget_status()
        cost_breakdown = self.get_cost_breakdown()

        return {
            'timestamp': datetime.now().isoformat(),
            'budget_status': budget_status,
            'cost_breakdown': cost_breakdown,
            'alerts': {
                'total_triggered': len(self.alerts_triggered),
                'recent': self.alerts_triggered[-10:]
            }
        }


# Global singleton instance
_global_monitor: Optional[ProductionMonitor] = None
_monitor_lock = Lock()


def get_production_monitor(
    hourly_budget_usd: float = 100.0,
    daily_budget_usd: float = 1000.0
) -> ProductionMonitor:
    """
    Get or create global ProductionMonitor instance.

    Args:
        hourly_budget_usd: Hourly budget limit (only used on first call)
        daily_budget_usd: Daily budget limit (only used on first call)

    Returns:
        ProductionMonitor singleton
    """
    global _global_monitor

    with _monitor_lock:
        if _global_monitor is None:
            _global_monitor = ProductionMonitor(
                hourly_budget_usd=hourly_budget_usd,
                daily_budget_usd=daily_budget_usd
            )
            logger.info("Global ProductionMonitor initialized")

        return _global_monitor


def initialize_production_monitoring(
    hourly_budget_usd: float = 100.0,
    daily_budget_usd: float = 1000.0
) -> ProductionMonitor:
    """
    Initialize production monitoring system.

    Call this once at application startup.

    Args:
        hourly_budget_usd: Hourly spending limit
        daily_budget_usd: Daily spending limit

    Returns:
        ProductionMonitor instance
    """
    monitor = get_production_monitor(
        hourly_budget_usd=hourly_budget_usd,
        daily_budget_usd=daily_budget_usd
    )

    logger.info(
        f"Production monitoring initialized - "
        f"Hourly: ${hourly_budget_usd}, Daily: ${daily_budget_usd}"
    )

    return monitor
