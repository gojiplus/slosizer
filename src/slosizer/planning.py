"""Capacity planning algorithms.

This module provides functions to determine optimal reserved capacity
based on throughput or latency targets.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from slosizer._utils import round_up_to_increment
from slosizer.schema import (
    CapacityProfile,
    LatencyTarget,
    PlanOptions,
    PlanResult,
    RequestTrace,
    ThroughputTarget,
)
from slosizer.simulation import bucket_required_units, simulate_capacity, summarize_slack


def _candidate_units(min_units: int, purchase_increment: int, max_units: int) -> list[int]:
    """Generate candidate unit counts for capacity search."""
    return list(range(min_units, max_units + 1, purchase_increment))


def _flatten_slack_summary(slack_summary: pd.DataFrame) -> dict[str, float]:
    """Flatten slack summary into a metrics dictionary."""
    metrics: dict[str, float] = {}
    if slack_summary.empty:
        return metrics
    for row in slack_summary.itertuples(index=False):
        suffix = (
            str(int(float(row.window_s))) if float(row.window_s).is_integer() else str(row.window_s)
        )
        metrics[f"avg_spare_units_{suffix}s"] = float(row.avg_spare_units)
        metrics[f"avg_spare_fraction_{suffix}s"] = float(row.avg_spare_fraction)
        metrics[f"overload_probability_{suffix}s"] = float(row.overload_probability)
        metrics[f"p99_required_units_{suffix}s"] = float(row.p99_required_units)
    metrics["worst_window_overload_probability"] = float(
        slack_summary["overload_probability"].max()
    )
    return metrics


def _plan_throughput(
    trace: RequestTrace,
    profile: CapacityProfile,
    target: ThroughputTarget,
    options: PlanOptions,
) -> PlanResult:
    """Plan capacity for a throughput target."""
    base_table = bucket_required_units(
        trace.frame,
        profile,
        units=0,
        windows_s=target.windows_s,
        output_token_source=options.output_token_source,
    )

    floor_units = float(profile.min_units)
    if target.percentile is not None:
        quantiles = base_table.groupby("window_s")["required_units"].quantile(target.percentile)
        floor_units = max(floor_units, float(quantiles.max()))

    floor_units = float(
        round_up_to_increment(floor_units, profile.min_units, profile.purchase_increment)
    )
    floor_units = float(
        round_up_to_increment(
            floor_units * (1.0 + options.headroom_factor),
            profile.min_units,
            profile.purchase_increment,
        )
    )

    recommended = None
    slack_summary = pd.DataFrame()
    for units in _candidate_units(
        int(floor_units), profile.purchase_increment, options.max_units_to_search
    ):
        table = bucket_required_units(
            trace.frame,
            profile,
            units=units,
            windows_s=target.windows_s,
            output_token_source=options.output_token_source,
        )
        summary = summarize_slack(table)
        if target.max_overload_probability is None:
            recommended = units
            slack_summary = summary
            break
        if float(summary["overload_probability"].max()) <= target.max_overload_probability:
            recommended = units
            slack_summary = summary
            break

    if recommended is None:
        raise RuntimeError(
            f"No feasible throughput plan found. "
            f"Target: {target.label()}, "
            f"search range: {profile.min_units}-{options.max_units_to_search} {profile.unit_name}. "
            f"Try increasing max_units_to_search or relaxing the target "
            f"(higher percentile or max_overload_probability)."
        )

    metrics: dict[str, float | str] = {
        "planning_percentile": float(target.percentile)
        if target.percentile is not None
        else float("nan"),
        "max_overload_probability_target": (
            float(target.max_overload_probability)
            if target.max_overload_probability is not None
            else float("nan")
        ),
    }
    metrics.update(_flatten_slack_summary(slack_summary))
    assumptions = {
        "output_token_source": options.output_token_source,
        "windows_s": target.windows_s,
        "headroom_factor": options.headroom_factor,
    }
    return PlanResult(
        objective="throughput",
        target=target.label(),
        recommended_units=int(recommended),
        unit_name=profile.unit_name,
        metrics=metrics,
        slack_summary=slack_summary,
        assumptions=assumptions,
    )


def _plan_latency(
    trace: RequestTrace,
    profile: CapacityProfile,
    target: LatencyTarget,
    options: PlanOptions,
) -> PlanResult:
    """Plan capacity for a latency target."""
    metric_col = "total_latency_s" if str(target.slo.metric) == "e2e" else "queue_delay_s"
    recommended_base = None

    for units in _candidate_units(
        profile.min_units, profile.purchase_increment, options.max_units_to_search
    ):
        simulation = simulate_capacity(trace, profile, units=units, options=options)
        quantile_value = float(
            np.quantile(simulation.request_level[metric_col], target.slo.percentile)
        )
        if quantile_value <= target.slo.threshold_s:
            recommended_base = units
            break

    if recommended_base is None:
        raise RuntimeError(
            f"No feasible latency plan found. "
            f"Target: p{int(target.slo.percentile * 100)} {target.slo.metric.value} <= {target.slo.threshold_s}s, "
            f"search range: {profile.min_units}-{options.max_units_to_search} {profile.unit_name}. "
            f"Try increasing max_units_to_search, raising threshold_s, "
            f"or lowering percentile."
        )

    recommended = round_up_to_increment(
        recommended_base * (1.0 + options.headroom_factor),
        profile.min_units,
        profile.purchase_increment,
    )
    simulation = simulate_capacity(trace, profile, units=recommended, options=options)
    final_quantile = float(np.quantile(simulation.request_level[metric_col], target.slo.percentile))
    latency_row = simulation.latency_summary.iloc[0].to_dict()
    metrics: dict[str, float | str] = {
        "achieved_latency_quantile_s": final_quantile,
        "latency_percentile": float(target.slo.percentile),
        "latency_threshold_s": float(target.slo.threshold_s),
        "latency_metric": target.slo.metric,
        **{key: float(value) for key, value in latency_row.items()},
        **_flatten_slack_summary(simulation.slack_summary),
    }
    assumptions = dict(simulation.assumptions)
    assumptions["headroom_factor"] = options.headroom_factor
    return PlanResult(
        objective="latency",
        target=target.label(),
        recommended_units=int(recommended),
        unit_name=profile.unit_name,
        metrics=metrics,
        slack_summary=simulation.slack_summary,
        latency_summary=simulation.latency_summary,
        request_level=simulation.request_level,
        assumptions=assumptions,
    )


def plan_capacity(
    trace: RequestTrace,
    profile: CapacityProfile,
    target: ThroughputTarget | LatencyTarget,
    *,
    options: PlanOptions | None = None,
) -> PlanResult:
    """Determine optimal reserved capacity for a target.

    Searches over candidate capacity levels to find the minimum that
    satisfies the given throughput or latency target.

    Args:
        trace: Request trace representing workload.
        profile: Capacity profile for the target provider/model.
        target: Throughput or latency target to meet.
        options: Planning options.

    Returns:
        PlanResult with recommended capacity and metrics.

    Raises:
        ValueError: If profile.throughput_per_unit is not set.
        TypeError: If target is not a ThroughputTarget or LatencyTarget.
    """
    if options is None:
        options = PlanOptions()

    if profile.throughput_per_unit is None:
        raise ValueError("profile.throughput_per_unit must be set before planning.")

    if isinstance(target, ThroughputTarget):
        return _plan_throughput(trace, profile, target, options)
    if isinstance(target, LatencyTarget):
        return _plan_latency(trace, profile, target, options)
    raise TypeError("target must be a ThroughputTarget or LatencyTarget instance.")


def compare_scenarios(
    scenarios: Mapping[str, RequestTrace],
    profile: CapacityProfile,
    targets: Sequence[ThroughputTarget | LatencyTarget],
    *,
    options: PlanOptions | None = None,
) -> pd.DataFrame:
    """Compare capacity requirements across scenarios and targets.

    Args:
        scenarios: Named request traces to compare.
        profile: Capacity profile for planning.
        targets: Throughput and/or latency targets.
        options: Planning options.

    Returns:
        DataFrame with planning results for each scenario/target combination.
    """
    if options is None:
        options = PlanOptions()

    rows = []
    for scenario_name, trace in scenarios.items():
        for target in targets:
            result = plan_capacity(trace, profile, target, options=options)
            row = result.as_dict()
            row["scenario"] = scenario_name
            rows.append(row)
    return (
        pd.DataFrame(rows).sort_values(["scenario", "objective", "target"]).reset_index(drop=True)
    )
