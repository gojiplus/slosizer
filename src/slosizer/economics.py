"""Optional absolute-profit and cross-model planning.

The economic planner keeps business value and SLO policy separate from provider
capacity facts and rate cards. It evaluates every legal provisioned-capacity
choice and maximizes expected hourly contribution after inference and SLO costs.
For the common fixed-demand case with a hard SLO, use hybrid cost optimization;
gross value is constant and does not need to be supplied.
"""

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from slosizer._utils import capacity_candidates
from slosizer.schema import (
    CapacityProfile,
    PlanOptions,
    ProfitPlanResult,
    ProfitScenario,
    ProfitTarget,
    RateCard,
    RequestTrace,
)
from slosizer.simulation import fit_baseline_latency_model, simulate_capacity


def _trace_duration_hours(trace: RequestTrace) -> float:
    """Return the positive observation span in hours."""
    arrivals = trace.frame["arrival_s"].to_numpy(dtype=float)
    if len(arrivals) < 2:
        raise ValueError("profit planning requires at least two requests")
    duration_s = float(arrivals.max() - arrivals.min())
    if duration_s <= 0:
        raise ValueError("profit planning requires a positive trace duration")
    return duration_s / 3600.0


def _gross_value(trace: RequestTrace, target: ProfitTarget) -> float:
    """Return total expected gross business value represented by a trace."""
    if target.value_per_request is not None:
        return target.value_per_request * len(trace.frame)

    if "business_value" not in trace.frame:
        raise ValueError(
            "value_per_request is required when the trace has no business_value column"
        )
    values = trace.frame["business_value"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "business_value must be complete and finite when value_per_request is omitted"
        )
    if (values < 0).any():
        raise ValueError("business_value must be non-negative")
    return float(values.sum())


def _candidate_units(profile: CapacityProfile, max_units: int) -> list[int]:
    """Return legal nonzero provisioned-capacity choices."""
    return capacity_candidates(profile.min_units, profile.purchase_increment, max_units)


def plan_profit_capacity(
    trace: RequestTrace,
    profile: CapacityProfile,
    pricing: RateCard,
    target: ProfitTarget,
    *,
    options: PlanOptions | None = None,
) -> ProfitPlanResult:
    """Choose reserved capacity that maximizes expected hourly profit.

    Profit is expected gross business value minus provisioned inference cost and,
    for a priced SLO, the expected cost of requests that miss the latency
    threshold. A hard SLO instead removes noncompliant capacity choices.

    The function intentionally does not apply ``headroom_factor`` after finding
    an optimum, because doing so would no longer maximize the stated objective.
    Forecast uncertainty should be represented by workload scenarios.

    Args:
        trace: Model-specific request trace or replay scenario.
        profile: Capacity facts for the model and deployment.
        pricing: Effective rate card. Only its provisioned price is used here.
        target: Business value and SLO policy.
        options: Simulation and search options. ``headroom_factor`` must be zero.

    Returns:
        Recommended capacity with the complete candidate frontier.

    Raises:
        ValueError: If required economic inputs are absent or inconsistent.
        RuntimeError: If no searched capacity choice satisfies a hard SLO.
    """
    if options is None:
        options = PlanOptions()
    if options.headroom_factor != 0:
        raise ValueError(
            "headroom_factor is not compatible with profit optimization; "
            "represent forecast uncertainty with scenarios"
        )
    if profile.throughput_per_unit is None:
        raise ValueError("profile.throughput_per_unit must be set before planning")
    pricing.validate_for(profile)

    duration_h = _trace_duration_hours(trace)
    gross_value_hourly = _gross_value(trace, target) / duration_h
    baseline_model = options.baseline_latency_model or fit_baseline_latency_model(trace)
    simulation_options = replace(options, baseline_latency_model=baseline_model)
    metric_col = (
        "total_latency_s"
        if str(target.latency_slo.metric) == "e2e"
        else "queue_delay_s"
    )

    rows: list[dict[str, Any]] = []
    for units in _candidate_units(profile, options.max_units_to_search):
        simulation = simulate_capacity(
            trace, profile, units=units, options=simulation_options
        )
        latency = simulation.request_level[metric_col].to_numpy(dtype=float)
        bad_requests = int((latency > target.latency_slo.threshold_s).sum())
        slo_attainment = 1.0 - bad_requests / len(latency)
        meets_slo = slo_attainment >= target.latency_slo.percentile
        provisioned_cost_hourly = units * pricing.provisioned.cost_per_unit_hour
        slo_cost_hourly = (
            bad_requests * target.slo_violation_cost_per_request / duration_h
            if target.slo_policy == "priced"
            else 0.0
        )
        rows.append(
            {
                "units": units,
                "gross_value_hourly": gross_value_hourly,
                "provisioned_cost_hourly": provisioned_cost_hourly,
                "slo_violation_cost_hourly": slo_cost_hourly,
                "expected_profit_hourly": (
                    gross_value_hourly - provisioned_cost_hourly - slo_cost_hourly
                ),
                "slo_attainment": slo_attainment,
                "bad_requests": bad_requests,
                "meets_slo": meets_slo,
            }
        )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise RuntimeError(
            f"No legal capacity choices at or below {options.max_units_to_search} "
            f"{profile.unit_name}"
        )
    eligible = (
        candidates[candidates["meets_slo"]]
        if target.slo_policy == "hard"
        else candidates
    )
    if eligible.empty:
        raise RuntimeError(
            "No capacity choice satisfies the hard latency SLO within the search range"
        )
    best = (
        eligible.sort_values("units", kind="stable")
        .sort_values("expected_profit_hourly", ascending=False, kind="stable")
        .iloc[0]
    )

    assumptions: dict[str, Any] = {
        "provider": profile.provider,
        "model": profile.model,
        "deployment_type": profile.deployment_type,
        "region": profile.region,
        "currency": pricing.currency,
        "pricing_effective_from": pricing.effective_from,
        "pricing_effective_to": pricing.effective_to,
        "pricing_verified_on": pricing.verified_on,
        "pricing_source": pricing.source,
        "slo_policy": target.slo_policy,
        "slo_threshold_s": target.latency_slo.threshold_s,
        "slo_percentile": target.latency_slo.percentile,
        "slo_metric": str(target.latency_slo.metric),
        "trace_duration_hours": duration_h,
        "value_source": (
            "value_per_request"
            if target.value_per_request is not None
            else "business_value"
        ),
    }
    return ProfitPlanResult(
        recommended_units=int(best["units"]),
        unit_name=profile.unit_name,
        currency=pricing.currency,
        gross_value_hourly=float(best["gross_value_hourly"]),
        provisioned_cost_hourly=float(best["provisioned_cost_hourly"]),
        slo_violation_cost_hourly=float(best["slo_violation_cost_hourly"]),
        expected_profit_hourly=float(best["expected_profit_hourly"]),
        slo_attainment=float(best["slo_attainment"]),
        candidate_plans=candidates,
        assumptions=assumptions,
    )


def compare_profit_scenarios(
    scenarios: list[ProfitScenario], *, options: PlanOptions | None = None
) -> pd.DataFrame:
    """Compare externally forecast model scenarios by expected hourly profit.

    The function does not estimate demand effects. Each scenario's trace defines
    its request count, timing, token mix, and burstiness.
    """
    if not scenarios:
        raise ValueError("scenarios must not be empty")
    currencies = {scenario.pricing.currency for scenario in scenarios}
    if len(currencies) != 1:
        raise ValueError(
            "all profit scenarios must use the same currency before ranking"
        )

    rows = []
    for scenario in scenarios:
        result = plan_profit_capacity(
            scenario.trace,
            scenario.profile,
            scenario.pricing,
            scenario.target,
            options=options,
        )
        row = result.as_dict()
        row.update(
            {
                "scenario": scenario.name,
                "provider": scenario.profile.provider,
                "model": scenario.profile.model,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["expected_profit_hourly", "scenario"],
        ascending=[False, True],
        ignore_index=True,
    )
