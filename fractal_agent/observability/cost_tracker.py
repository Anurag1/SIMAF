"""
Cost Tracking System for Fractal Agent

Comprehensive LLM cost tracking with real-time monitoring, budget alerts,
and detailed breakdowns by tier, agent, provider, and model.

Features:
- Accurate per-token cost calculation using ModelRegistry
- Multi-dimensional cost aggregation (tier, agent, provider, model, time)
- Budget monitoring with configurable thresholds
- Cost projection and trend analysis
- Thread-safe operations for concurrent usage
- Export capabilities for dashboards and reporting
- Integration with Prometheus metrics

Usage:
    from fractal_agent.observability.cost_tracker import CostTracker, get_cost_tracker

    # Get singleton instance
    tracker = get_cost_tracker()

    # Record LLM call
    cost_info = tracker.record_call(
        tier="System1_Research",
        agent="ResearchAgent",
        provider="anthropic",
        model="claude-sonnet-4.5",
        input_tokens=1000,
        output_tokens=500,
        cache_hit=False
    )

    # Check budget status
    status = tracker.get_budget_status()

    # Get cost breakdown
    breakdown = tracker.get_cost_breakdown(hours=24)

Author: BMad
Date: 2025-01-20
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Lock
import json

from ..utils.model_registry import ModelRegistry, ModelInfo, get_registry

logger = logging.getLogger(__name__)


@dataclass
class CostEntry:
    """Single LLM call cost entry"""
    timestamp: datetime
    tier: str
    agent: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    cache_hit: bool
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class CostWindow:
    """Tracks costs within a time window for budget monitoring"""
    window_start: datetime
    window_duration_hours: int
    entries: List[CostEntry] = field(default_factory=list)
    total_cost: float = 0.0

    def add_entry(self, entry: CostEntry):
        """Add a cost entry"""
        self.entries.append(entry)
        self.total_cost += entry.total_cost_usd

    def prune_old_entries(self):
        """Remove entries outside the time window"""
        cutoff = datetime.now() - timedelta(hours=self.window_duration_hours)
        new_entries = [e for e in self.entries if e.timestamp >= cutoff]
        self.entries = new_entries
        self.total_cost = sum(e.total_cost_usd for e in new_entries)

    def get_rate_per_hour(self) -> float:
        """Get average cost per hour in current window"""
        self.prune_old_entries()
        if not self.entries:
            return 0.0
        duration_hours = (datetime.now() - self.entries[0].timestamp).total_seconds() / 3600
        if duration_hours == 0:
            return 0.0
        return self.total_cost / duration_hours

    def get_entries_count(self) -> int:
        """Get number of entries in window"""
        self.prune_old_entries()
        return len(self.entries)


@dataclass
class BudgetAlert:
    """Budget alert record"""
    alert_type: str  # 'hourly' or 'daily'
    timestamp: datetime
    current_spend: float
    budget_limit: float
    overage_percent: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class CostTracker:
    """
    Comprehensive cost tracking system for LLM operations.

    Provides:
    - Real-time cost calculation with input/output token separation
    - Multi-dimensional cost aggregation (tier, agent, provider, model)
    - Time-windowed tracking (hourly, daily, weekly, monthly)
    - Budget monitoring with configurable alerts
    - Cost projection and trend analysis
    - Thread-safe operations
    - Export capabilities for dashboards
    """

    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        hourly_budget_usd: float = 100.0,
        daily_budget_usd: float = 1000.0,
        enable_detailed_tracking: bool = True
    ):
        """
        Initialize cost tracker.

        Args:
            model_registry: Model registry for pricing (uses global if None)
            hourly_budget_usd: Hourly spending limit for alerts
            daily_budget_usd: Daily spending limit for alerts
            enable_detailed_tracking: Keep detailed entry history (vs. aggregates only)
        """
        self.model_registry = model_registry or get_registry()
        self.hourly_budget_usd = hourly_budget_usd
        self.daily_budget_usd = daily_budget_usd
        self.enable_detailed_tracking = enable_detailed_tracking

        # Thread safety
        self._lock = Lock()

        # Cost tracking windows
        self.hourly_window = CostWindow(
            window_start=datetime.now(),
            window_duration_hours=1
        )
        self.daily_window = CostWindow(
            window_start=datetime.now(),
            window_duration_hours=24
        )
        self.weekly_window = CostWindow(
            window_start=datetime.now(),
            window_duration_hours=168  # 7 days
        )

        # Aggregated cost breakdowns
        self.cost_by_tier: Dict[str, float] = defaultdict(float)
        self.cost_by_agent: Dict[str, float] = defaultdict(float)
        self.cost_by_provider: Dict[str, float] = defaultdict(float)
        self.cost_by_model: Dict[str, float] = defaultdict(float)

        # Token usage tracking
        self.tokens_by_tier: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'input': 0, 'output': 0}
        )
        self.tokens_by_model: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'input': 0, 'output': 0}
        )

        # Performance tracking
        self.call_count_by_tier: Dict[str, int] = defaultdict(int)
        self.cache_hits_by_tier: Dict[str, int] = defaultdict(int)
        self.total_latency_by_tier: Dict[str, float] = defaultdict(float)

        # Alert tracking
        self.alerts: List[BudgetAlert] = []
        self.last_hourly_alert: Optional[datetime] = None
        self.last_daily_alert: Optional[datetime] = None

        # Statistics
        self.start_time = datetime.now()
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0

        logger.info(
            f"CostTracker initialized - "
            f"Hourly budget: ${hourly_budget_usd}, Daily budget: ${daily_budget_usd}"
        )

    def record_call(
        self,
        tier: str,
        agent: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_hit: bool = False,
        latency_ms: float = 0.0
    ) -> Dict[str, Any]:
        """
        Record an LLM call and calculate costs.

        Args:
            tier: VSM tier (e.g., "System1_Research")
            agent: Agent name (e.g., "ResearchAgent")
            provider: Provider name (e.g., "anthropic")
            model: Model ID (e.g., "claude-sonnet-4.5")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cache_hit: Whether this was a cache hit
            latency_ms: Call latency in milliseconds

        Returns:
            Dict with cost breakdown and tracking info
        """
        with self._lock:
            # Get model info for accurate pricing
            model_info = self.model_registry.get_model_info(model)

            if model_info is None:
                logger.warning(f"Model {model} not found in registry, using defaults")
                input_cost_per_mtok = 3.0
                output_cost_per_mtok = 15.0
            else:
                input_cost_per_mtok = model_info.input_cost_per_mtok
                output_cost_per_mtok = model_info.output_cost_per_mtok

            # Calculate costs
            input_cost = (input_tokens / 1_000_000) * input_cost_per_mtok
            output_cost = (output_tokens / 1_000_000) * output_cost_per_mtok

            # Apply cache discount (90% off for cached reads)
            if cache_hit:
                input_cost *= 0.1
                output_cost *= 0.1

            total_cost = input_cost + output_cost

            # Create cost entry
            entry = CostEntry(
                timestamp=datetime.now(),
                tier=tier,
                agent=agent,
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost_usd=input_cost,
                output_cost_usd=output_cost,
                total_cost_usd=total_cost,
                cache_hit=cache_hit,
                latency_ms=latency_ms
            )

            # Add to windows
            self.hourly_window.add_entry(entry)
            self.daily_window.add_entry(entry)
            self.weekly_window.add_entry(entry)

            # Update aggregations
            self._update_aggregations(entry)

            # Check budget alerts
            self._check_budget_alerts()

            # Update statistics
            self.total_calls += 1
            self.total_tokens += input_tokens + output_tokens
            self.total_cost += total_cost

            return {
                'input_cost_usd': input_cost,
                'output_cost_usd': output_cost,
                'total_cost_usd': total_cost,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'cache_hit': cache_hit,
                'cumulative_cost_usd': self.total_cost,
                'hourly_rate_usd': self.hourly_window.get_rate_per_hour(),
                'daily_total_usd': self.daily_window.total_cost
            }

    def _update_aggregations(self, entry: CostEntry):
        """Update all cost aggregations"""
        # Cost breakdowns
        self.cost_by_tier[entry.tier] += entry.total_cost_usd
        self.cost_by_agent[entry.agent] += entry.total_cost_usd
        self.cost_by_provider[entry.provider] += entry.total_cost_usd
        model_key = f"{entry.provider}/{entry.model}"
        self.cost_by_model[model_key] += entry.total_cost_usd

        # Token tracking
        self.tokens_by_tier[entry.tier]['input'] += entry.input_tokens
        self.tokens_by_tier[entry.tier]['output'] += entry.output_tokens
        self.tokens_by_model[model_key]['input'] += entry.input_tokens
        self.tokens_by_model[model_key]['output'] += entry.output_tokens

        # Performance tracking
        self.call_count_by_tier[entry.tier] += 1
        if entry.cache_hit:
            self.cache_hits_by_tier[entry.tier] += 1
        self.total_latency_by_tier[entry.tier] += entry.latency_ms

    def _check_budget_alerts(self):
        """Check if budget thresholds are exceeded"""
        now = datetime.now()
        hourly_rate = self.hourly_window.get_rate_per_hour()
        daily_total = self.daily_window.total_cost

        # Hourly budget alert (throttle to once per hour)
        if hourly_rate > self.hourly_budget_usd:
            if (self.last_hourly_alert is None or
                (now - self.last_hourly_alert) > timedelta(hours=1)):

                overage_pct = ((hourly_rate / self.hourly_budget_usd) - 1) * 100
                alert = BudgetAlert(
                    alert_type='hourly',
                    timestamp=now,
                    current_spend=hourly_rate,
                    budget_limit=self.hourly_budget_usd,
                    overage_percent=overage_pct,
                    message=f"Hourly budget exceeded: ${hourly_rate:.2f}/hr "
                            f"(budget: ${self.hourly_budget_usd:.2f}/hr, "
                            f"+{overage_pct:.1f}% over)"
                )
                self.alerts.append(alert)
                self.last_hourly_alert = now
                logger.warning(alert.message)

        # Daily budget alert (throttle to once per 6 hours)
        if daily_total > self.daily_budget_usd:
            if (self.last_daily_alert is None or
                (now - self.last_daily_alert) > timedelta(hours=6)):

                overage_pct = ((daily_total / self.daily_budget_usd) - 1) * 100
                alert = BudgetAlert(
                    alert_type='daily',
                    timestamp=now,
                    current_spend=daily_total,
                    budget_limit=self.daily_budget_usd,
                    overage_percent=overage_pct,
                    message=f"Daily budget exceeded: ${daily_total:.2f}/day "
                            f"(budget: ${self.daily_budget_usd:.2f}/day, "
                            f"+{overage_pct:.1f}% over)"
                )
                self.alerts.append(alert)
                self.last_daily_alert = now
                logger.warning(alert.message)

    def get_budget_status(self) -> Dict[str, Any]:
        """
        Get current budget status with utilization details.

        Returns:
            Dict with budget metrics and alert status
        """
        with self._lock:
            self.hourly_window.prune_old_entries()
            self.daily_window.prune_old_entries()

            hourly_rate = self.hourly_window.get_rate_per_hour()
            daily_total = self.daily_window.total_cost

            return {
                'hourly': {
                    'rate_usd': hourly_rate,
                    'budget_usd': self.hourly_budget_usd,
                    'utilization_pct': (hourly_rate / self.hourly_budget_usd) * 100 if self.hourly_budget_usd > 0 else 0,
                    'remaining_usd': max(0, self.hourly_budget_usd - hourly_rate),
                    'exceeded': hourly_rate > self.hourly_budget_usd
                },
                'daily': {
                    'total_usd': daily_total,
                    'budget_usd': self.daily_budget_usd,
                    'utilization_pct': (daily_total / self.daily_budget_usd) * 100 if self.daily_budget_usd > 0 else 0,
                    'remaining_usd': max(0, self.daily_budget_usd - daily_total),
                    'exceeded': daily_total > self.daily_budget_usd
                },
                'alert_triggered': (
                    hourly_rate > self.hourly_budget_usd or
                    daily_total > self.daily_budget_usd
                ),
                'total_alerts': len(self.alerts),
                'recent_alerts': [a.to_dict() for a in self.alerts[-5:]]
            }

    def get_cost_breakdown(
        self,
        hours: Optional[int] = None,
        by_dimension: str = 'all'
    ) -> Dict[str, Any]:
        """
        Get cost breakdown by various dimensions.

        Args:
            hours: Time window in hours (None for all-time)
            by_dimension: 'tier', 'agent', 'provider', 'model', or 'all'

        Returns:
            Dict with cost breakdown
        """
        with self._lock:
            # Select time window
            if hours is None:
                total_cost = self.total_cost
                window_desc = 'all-time'
            elif hours <= 1:
                self.hourly_window.prune_old_entries()
                total_cost = self.hourly_window.total_cost
                window_desc = 'last hour'
            elif hours <= 24:
                self.daily_window.prune_old_entries()
                total_cost = self.daily_window.total_cost
                window_desc = 'last 24 hours'
            else:
                self.weekly_window.prune_old_entries()
                total_cost = self.weekly_window.total_cost
                window_desc = 'last 7 days'

            result = {
                'time_window': window_desc,
                'time_window_hours': hours,
                'total_cost_usd': total_cost,
                'timestamp': datetime.now().isoformat()
            }

            # Add requested dimensions
            if by_dimension in ('tier', 'all'):
                result['by_tier'] = dict(self.cost_by_tier)

            if by_dimension in ('agent', 'all'):
                result['by_agent'] = dict(self.cost_by_agent)

            if by_dimension in ('provider', 'all'):
                result['by_provider'] = dict(self.cost_by_provider)

            if by_dimension in ('model', 'all'):
                result['by_model'] = dict(self.cost_by_model)

            return result

    def get_token_breakdown(self) -> Dict[str, Any]:
        """
        Get token usage breakdown by tier and model.

        Returns:
            Dict with token usage statistics
        """
        with self._lock:
            return {
                'total_tokens': self.total_tokens,
                'total_calls': self.total_calls,
                'avg_tokens_per_call': self.total_tokens / max(1, self.total_calls),
                'by_tier': dict(self.tokens_by_tier),
                'by_model': dict(self.tokens_by_model),
                'timestamp': datetime.now().isoformat()
            }

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics by tier.

        Returns:
            Dict with performance metrics
        """
        with self._lock:
            stats_by_tier = {}

            for tier in self.call_count_by_tier.keys():
                call_count = self.call_count_by_tier[tier]
                cache_hits = self.cache_hits_by_tier[tier]
                total_latency = self.total_latency_by_tier[tier]

                stats_by_tier[tier] = {
                    'total_calls': call_count,
                    'cache_hits': cache_hits,
                    'cache_hit_rate': cache_hits / max(1, call_count),
                    'avg_latency_ms': total_latency / max(1, call_count),
                    'total_cost_usd': self.cost_by_tier[tier]
                }

            return {
                'by_tier': stats_by_tier,
                'overall': {
                    'total_calls': self.total_calls,
                    'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                    'calls_per_hour': self.total_calls / max(0.01, (datetime.now() - self.start_time).total_seconds() / 3600)
                },
                'timestamp': datetime.now().isoformat()
            }

    def get_cost_projection(
        self,
        projection_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Project future costs based on current usage rate.

        Args:
            projection_hours: Hours to project forward

        Returns:
            Dict with cost projections
        """
        with self._lock:
            hourly_rate = self.hourly_window.get_rate_per_hour()

            return {
                'current_hourly_rate_usd': hourly_rate,
                'projection_hours': projection_hours,
                'projected_cost_usd': hourly_rate * projection_hours,
                'projected_daily_cost_usd': hourly_rate * 24,
                'projected_weekly_cost_usd': hourly_rate * 168,
                'projected_monthly_cost_usd': hourly_rate * 730,
                'timestamp': datetime.now().isoformat()
            }

    def export_summary(self) -> Dict[str, Any]:
        """
        Export comprehensive summary for dashboards.

        Returns:
            Dict with all tracking data
        """
        return {
            'budget_status': self.get_budget_status(),
            'cost_breakdown': self.get_cost_breakdown(),
            'token_breakdown': self.get_token_breakdown(),
            'performance_stats': self.get_performance_stats(),
            'cost_projection': self.get_cost_projection(),
            'timestamp': datetime.now().isoformat()
        }

    def reset(self):
        """Reset all tracking data (use carefully - for testing only)"""
        with self._lock:
            self.hourly_window = CostWindow(
                window_start=datetime.now(),
                window_duration_hours=1
            )
            self.daily_window = CostWindow(
                window_start=datetime.now(),
                window_duration_hours=24
            )
            self.weekly_window = CostWindow(
                window_start=datetime.now(),
                window_duration_hours=168
            )

            self.cost_by_tier.clear()
            self.cost_by_agent.clear()
            self.cost_by_provider.clear()
            self.cost_by_model.clear()

            self.tokens_by_tier.clear()
            self.tokens_by_model.clear()

            self.call_count_by_tier.clear()
            self.cache_hits_by_tier.clear()
            self.total_latency_by_tier.clear()

            self.alerts.clear()
            self.last_hourly_alert = None
            self.last_daily_alert = None

            self.start_time = datetime.now()
            self.total_calls = 0
            self.total_tokens = 0
            self.total_cost = 0.0

            logger.info("CostTracker reset")


# Global singleton instance
_global_tracker: Optional[CostTracker] = None
_tracker_lock = Lock()


def get_cost_tracker(
    hourly_budget_usd: float = 100.0,
    daily_budget_usd: float = 1000.0
) -> CostTracker:
    """
    Get or create global CostTracker instance.

    Args:
        hourly_budget_usd: Hourly budget limit (only used on first call)
        daily_budget_usd: Daily budget limit (only used on first call)

    Returns:
        CostTracker singleton
    """
    global _global_tracker

    with _tracker_lock:
        if _global_tracker is None:
            _global_tracker = CostTracker(
                hourly_budget_usd=hourly_budget_usd,
                daily_budget_usd=daily_budget_usd
            )
            logger.info("Global CostTracker initialized")

        return _global_tracker


def initialize_cost_tracking(
    hourly_budget_usd: float = 100.0,
    daily_budget_usd: float = 1000.0
) -> CostTracker:
    """
    Initialize cost tracking system.

    Call this once at application startup.

    Args:
        hourly_budget_usd: Hourly spending limit
        daily_budget_usd: Daily spending limit

    Returns:
        CostTracker instance
    """
    tracker = get_cost_tracker(
        hourly_budget_usd=hourly_budget_usd,
        daily_budget_usd=daily_budget_usd
    )

    logger.info(
        f"Cost tracking initialized - "
        f"Hourly: ${hourly_budget_usd}, Daily: ${daily_budget_usd}"
    )

    return tracker
