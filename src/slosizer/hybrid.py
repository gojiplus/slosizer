"""Hybrid provisioned + paygo capacity planning.

This module implements hybrid capacity planning that combines provisioned capacity
for baseline load with pay-as-you-go for peak overflow traffic.
"""

from __future__ import annotations

import numpy as np

from slosizer._utils import round_up_to_increment
from slosizer.schema import (
    CapacityProfile,
    HybridPlanResult,
    HybridPricingModel,
    HybridTarget,
    LatencySLO,
    PlanOptions,
    RequestTrace,
)
from slosizer.simulation import bucket_with_tokens, simulate_capacity, summarize_slack


def compute_overflow_tokens(
    trace: RequestTrace,
    profile: CapacityProfile,
    *,
    units: int,
    window_s: float = 1.0,
    output_token_source: str = "observed",
) -> dict[str, float]:
    """Compute overflow token counts for a given provisioned capacity level.

    Args:
        trace: Request trace to analyze.
        profile: Capacity profile with throughput settings.
        units: Number of provisioned capacity units.
        window_s: Time window size for bucket analysis.
        output_token_source: Source for output tokens.

    Returns:
        Dictionary with overflow metrics:
        - overflow_input_tokens: Total overflow input tokens.
        - overflow_output_tokens: Total overflow output tokens.
        - overflow_fraction: Fraction of buckets with overflow.
        - trace_duration_hours: Duration of trace in hours.
    """
    buckets = bucket_with_tokens(
        trace.frame,
        profile,
        units=units,
        window_s=window_s,
        output_token_source=output_token_source,
    )

    if buckets.empty:
        return {
            "overflow_input_tokens": 0.0,
            "overflow_output_tokens": 0.0,
            "overflow_fraction": 0.0,
            "trace_duration_hours": 0.0,
        }

    overflow_frac = buckets["overflow_fraction"].to_numpy()
    overflow_input = (buckets["input_tokens"].to_numpy() * overflow_frac).sum()
    overflow_output = (buckets["output_tokens"].to_numpy() * overflow_frac).sum()
    overflow_bucket_fraction = float((overflow_frac > 0).mean())

    arrivals = trace.frame["arrival_s"]
    trace_duration_hours = max(1e-9, float(arrivals.max() - arrivals.min())) / 3600.0

    return {
        "overflow_input_tokens": float(overflow_input),
        "overflow_output_tokens": float(overflow_output),
        "overflow_fraction": overflow_bucket_fraction,
        "trace_duration_hours": trace_duration_hours,
    }


def _compute_hybrid_cost(
    trace: RequestTrace,
    profile: CapacityProfile,
    pricing: HybridPricingModel,
    *,
    units: int,
    window_s: float,
    output_token_source: str,
) -> dict[str, float]:
    """Compute hybrid costs for a given provisioned capacity level.

    Args:
        trace: Request trace to analyze.
        profile: Capacity profile with throughput settings.
        pricing: Hybrid pricing model with provisioned and paygo costs.
        units: Number of provisioned capacity units.
        window_s: Time window size for bucket analysis.
        output_token_source: Source for output tokens.

    Returns:
        Dictionary with cost breakdown.
    """
    overflow = compute_overflow_tokens(
        trace,
        profile,
        units=units,
        window_s=window_s,
        output_token_source=output_token_source,
    )

    provisioned_cost_hourly = float(units) * pricing.provisioned.cost_per_unit_hour

    duration_h = overflow["trace_duration_hours"]
    if duration_h > 0:
        overflow_input_hourly = overflow["overflow_input_tokens"] / duration_h
        overflow_output_hourly = overflow["overflow_output_tokens"] / duration_h
    else:
        overflow_input_hourly = 0.0
        overflow_output_hourly = 0.0

    paygo_cost_hourly = (
        overflow_input_hourly * pricing.paygo.input_cost_per_million / 1_000_000
        + overflow_output_hourly * pricing.paygo.output_cost_per_million / 1_000_000
    )

    return {
        "provisioned_cost_hourly": provisioned_cost_hourly,
        "paygo_cost_hourly": paygo_cost_hourly,
        "total_cost_hourly": provisioned_cost_hourly + paygo_cost_hourly,
        "overflow_fraction": overflow["overflow_fraction"],
        "overflow_input_tokens_hourly": overflow_input_hourly,
        "overflow_output_tokens_hourly": overflow_output_hourly,
    }


def _check_latency_slo(
    trace: RequestTrace,
    profile: CapacityProfile,
    slo: LatencySLO,
    *,
    units: int,
    options: PlanOptions,
) -> bool:
    """Check if a capacity level meets the latency SLO."""
    simulation = simulate_capacity(trace, profile, units=units, options=options)
    metric_col = "total_latency_s" if str(slo.metric) == "e2e" else "queue_delay_s"
    quantile_value = float(
        np.quantile(simulation.request_level[metric_col], slo.percentile)
    )
    return quantile_value <= slo.threshold_s


def _find_full_provision_units(
    trace: RequestTrace,
    profile: CapacityProfile,
    *,
    window_s: float,
    output_token_source: str,
) -> int:
    """Find minimum units needed to avoid any overflow."""
    buckets = bucket_with_tokens(
        trace.frame,
        profile,
        units=0,
        window_s=window_s,
        output_token_source=output_token_source,
    )

    if buckets.empty:
        return profile.min_units

    max_required = float(buckets["required_units"].max())
    return round_up_to_increment(
        max_required, profile.min_units, profile.purchase_increment
    )


def plan_hybrid_capacity(
    trace: RequestTrace,
    profile: CapacityProfile,
    pricing: HybridPricingModel,
    target: HybridTarget,
    options: PlanOptions | None = None,
) -> HybridPlanResult:
    """Find optimal hybrid provisioned + paygo capacity blend.

    Two strategies:
    1. cost_optimal: Sweep over provisioned levels, find minimum total cost.
    2. percentile_split: Provision at specified percentile, compute paygo cost.

    If latency_slo is specified, only considers configurations meeting the SLO.

    Args:
        trace: Request trace representing workload.
        profile: Capacity profile for the target provider/model.
        pricing: Hybrid pricing model with provisioned and paygo costs.
        target: Hybrid planning target (strategy and optional SLO).
        options: Planning options.

    Returns:
        HybridPlanResult with recommended hybrid configuration.

    Raises:
        ValueError: If profile.throughput_per_unit is not set.
        RuntimeError: If no feasible plan found (all configurations violate SLO).
    """
    if options is None:
        options = PlanOptions()

    if profile.throughput_per_unit is None:
        raise ValueError("profile.throughput_per_unit must be set before planning.")

    window_s = 1.0
    output_token_source = str(options.output_token_source)

    full_provision_units = _find_full_provision_units(
        trace,
        profile,
        window_s=window_s,
        output_token_source=output_token_source,
    )
    full_provision_cost = full_provision_units * pricing.provisioned.cost_per_unit_hour

    if target.strategy == "percentile_split":
        provision_percentile = target.provision_percentile
        if provision_percentile is None:
            raise ValueError(
                "provision_percentile is required when strategy='percentile_split'"
            )
        buckets = bucket_with_tokens(
            trace.frame,
            profile,
            units=0,
            window_s=window_s,
            output_token_source=output_token_source,
        )
        if buckets.empty:
            percentile_units = profile.min_units
        else:
            percentile_units = float(
                buckets["required_units"].quantile(provision_percentile)
            )
        provisioned_units = round_up_to_increment(
            percentile_units, profile.min_units, profile.purchase_increment
        )

        if target.latency_slo is not None and not _check_latency_slo(
            trace, profile, target.latency_slo, units=provisioned_units, options=options
        ):
            for units in range(
                provisioned_units + profile.purchase_increment,
                options.max_units_to_search + 1,
                profile.purchase_increment,
            ):
                if _check_latency_slo(
                    trace, profile, target.latency_slo, units=units, options=options
                ):
                    provisioned_units = units
                    break
            else:
                raise RuntimeError(
                    f"No feasible hybrid plan found. "
                    f"Latency SLO p{int(target.latency_slo.percentile * 100)} <= "
                    f"{target.latency_slo.threshold_s}s cannot be met within "
                    f"{options.max_units_to_search} {profile.unit_name}."
                )

        costs = _compute_hybrid_cost(
            trace,
            profile,
            pricing,
            units=provisioned_units,
            window_s=window_s,
            output_token_source=output_token_source,
        )

    else:
        best_units = 0
        best_cost = float("inf")
        best_costs: dict[str, float] = {}

        for units in range(
            0, options.max_units_to_search + 1, profile.purchase_increment
        ):
            effective_units = max(units, 0)

            if (
                target.latency_slo is not None
                and effective_units > 0
                and not _check_latency_slo(
                    trace,
                    profile,
                    target.latency_slo,
                    units=effective_units,
                    options=options,
                )
            ):
                continue

            costs = _compute_hybrid_cost(
                trace,
                profile,
                pricing,
                units=effective_units,
                window_s=window_s,
                output_token_source=output_token_source,
            )

            if costs["total_cost_hourly"] < best_cost:
                best_cost = costs["total_cost_hourly"]
                best_units = effective_units
                best_costs = costs

        if not best_costs:
            raise RuntimeError(
                f"No feasible hybrid plan found. "
                f"Search range: 0-{options.max_units_to_search} {profile.unit_name}. "
                f"Try increasing max_units_to_search or relaxing the latency SLO."
            )

        provisioned_units = best_units
        costs = best_costs

    headroom_units = round_up_to_increment(
        provisioned_units * (1.0 + options.headroom_factor),
        profile.min_units if provisioned_units > 0 else 0,
        profile.purchase_increment,
    )
    if headroom_units != provisioned_units:
        provisioned_units = headroom_units
        costs = _compute_hybrid_cost(
            trace,
            profile,
            pricing,
            units=provisioned_units,
            window_s=window_s,
            output_token_source=output_token_source,
        )

    from slosizer.simulation import bucket_required_units

    slack_table = bucket_required_units(
        trace.frame,
        profile,
        units=provisioned_units,
        windows_s=(1.0, 5.0, 30.0),
        output_token_source=output_token_source,
    )
    slack_summary = summarize_slack(slack_table)

    savings = full_provision_cost - costs["total_cost_hourly"]
    savings_percent = (
        (savings / full_provision_cost * 100) if full_provision_cost > 0 else 0.0
    )

    assumptions = {
        "strategy": target.strategy,
        "provision_percentile": target.provision_percentile,
        "output_token_source": output_token_source,
        "window_s": window_s,
        "headroom_factor": options.headroom_factor,
        "provisioned_cost_per_unit_hour": pricing.provisioned.cost_per_unit_hour,
        "paygo_input_cost_per_million": pricing.paygo.input_cost_per_million,
        "paygo_output_cost_per_million": pricing.paygo.output_cost_per_million,
    }
    if target.latency_slo is not None:
        assumptions["latency_slo_threshold_s"] = target.latency_slo.threshold_s
        assumptions["latency_slo_percentile"] = target.latency_slo.percentile
        assumptions["latency_slo_metric"] = str(target.latency_slo.metric)

    return HybridPlanResult(
        provisioned_units=provisioned_units,
        unit_name=profile.unit_name,
        provisioned_cost_hourly=costs["provisioned_cost_hourly"],
        paygo_cost_hourly=costs["paygo_cost_hourly"],
        total_cost_hourly=costs["total_cost_hourly"],
        full_provision_units=full_provision_units,
        full_provision_cost_hourly=full_provision_cost,
        savings_vs_full_provision=savings,
        savings_percent=savings_percent,
        overflow_fraction=costs["overflow_fraction"],
        overflow_input_tokens_hourly=costs["overflow_input_tokens_hourly"],
        overflow_output_tokens_hourly=costs["overflow_output_tokens_hourly"],
        slack_summary=slack_summary,
        assumptions=assumptions,
    )
