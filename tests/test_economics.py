"""Tests for profit-aware capacity planning."""

import pandas as pd
import pytest

from slosizer.economics import compare_profit_scenarios, plan_profit_capacity
from slosizer.ingest import from_dataframe
from slosizer.schema import (
    BaselineLatencyModel,
    CapacityProfile,
    LatencyMetric,
    LatencySLO,
    PlanOptions,
    ProfitScenario,
    ProfitTarget,
    ProvisionedPricing,
    RateCard,
    RequestSchema,
)


@pytest.fixture
def economic_trace():
    frame = pd.DataFrame(
        {
            "ts": [index * 0.5 for index in range(100)],
            "input_tokens": [100] * 100,
            "output_tokens": [0] * 100,
            "business_value": [1.0] * 100,
        }
    )
    return from_dataframe(frame, schema=RequestSchema())


@pytest.fixture
def economic_profile():
    return CapacityProfile(
        provider="test",
        model="model-a",
        unit_name="capacity_unit",
        throughput_per_unit=100.0,
    )


def pricing(cost_per_unit_hour: float = 1.0, currency: str = "USD") -> RateCard:
    return RateCard(
        provisioned=ProvisionedPricing(cost_per_unit_hour=cost_per_unit_hour),
        provider="test",
        model="model-a",
        currency=currency,
    )


def queue_slo() -> LatencySLO:
    return LatencySLO(
        threshold_s=0.1,
        percentile=0.95,
        metric=LatencyMetric.QUEUE_DELAY,
    )


def options() -> PlanOptions:
    return PlanOptions(
        max_units_to_search=5,
        baseline_latency_model=BaselineLatencyModel(
            intercept_s=0.0,
            input_token_s=0.0,
            cached_input_token_s=0.0,
            output_token_s=0.0,
            thinking_token_s=0.0,
        ),
    )


def test_hard_slo_selects_cheapest_compliant_capacity(economic_trace, economic_profile):
    result = plan_profit_capacity(
        economic_trace,
        economic_profile,
        pricing(),
        ProfitTarget(latency_slo=queue_slo()),
        options=options(),
    )

    assert result.recommended_units == 2
    assert result.slo_attainment >= 0.95
    assert result.provisioned_cost_hourly == 2.0
    assert len(result.candidate_plans) == 5


def test_priced_slo_can_rationally_accept_misses(economic_trace, economic_profile):
    result = plan_profit_capacity(
        economic_trace,
        economic_profile,
        pricing(),
        ProfitTarget(
            latency_slo=queue_slo(),
            slo_policy="priced",
            slo_violation_cost_per_request=0.0001,
        ),
        options=options(),
    )

    assert result.recommended_units == 1
    assert result.slo_attainment < 0.95
    assert result.slo_violation_cost_hourly > 0


def test_non_finite_simulated_latency_is_an_slo_miss(economic_trace, economic_profile):
    non_finite_model = BaselineLatencyModel(
        intercept_s=float("nan"),
        input_token_s=0.0,
        cached_input_token_s=0.0,
        output_token_s=0.0,
        thinking_token_s=0.0,
    )

    with pytest.raises(
        RuntimeError, match="No capacity choice satisfies the hard latency SLO"
    ):
        plan_profit_capacity(
            economic_trace,
            economic_profile,
            pricing(),
            ProfitTarget(latency_slo=LatencySLO(threshold_s=0.1, percentile=0.95)),
            options=PlanOptions(
                max_units_to_search=5,
                baseline_latency_model=non_finite_model,
            ),
        )


def test_scalar_value_can_replace_trace_value(economic_trace, economic_profile):
    trace = economic_trace
    trace.frame["business_value"] = float("nan")
    result = plan_profit_capacity(
        trace,
        economic_profile,
        pricing(),
        ProfitTarget(latency_slo=queue_slo(), value_per_request=2.0),
        options=options(),
    )

    assert result.gross_value_hourly == pytest.approx(200 / (49.5 / 3600))


def test_missing_business_value_rejected(economic_trace, economic_profile):
    economic_trace.frame["business_value"] = float("nan")
    with pytest.raises(ValueError, match="business_value must be complete"):
        plan_profit_capacity(
            economic_trace,
            economic_profile,
            pricing(),
            ProfitTarget(latency_slo=queue_slo()),
            options=options(),
        )


def test_headroom_is_not_silently_added(economic_trace, economic_profile):
    with pytest.raises(ValueError, match="headroom_factor"):
        plan_profit_capacity(
            economic_trace,
            economic_profile,
            pricing(),
            ProfitTarget(latency_slo=queue_slo()),
            options=PlanOptions(headroom_factor=0.1),
        )


def test_invalid_slo_policy_rejected():
    with pytest.raises(ValueError, match="unknown SLO policy"):
        ProfitTarget(latency_slo=queue_slo(), slo_policy="unknown")


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_value_per_request_rejected(value):
    with pytest.raises(ValueError, match="value_per_request must be finite"):
        ProfitTarget(latency_slo=queue_slo(), value_per_request=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_slo_violation_cost_rejected(value):
    with pytest.raises(
        ValueError, match="slo_violation_cost_per_request must be finite"
    ):
        ProfitTarget(
            latency_slo=queue_slo(),
            slo_policy="priced",
            slo_violation_cost_per_request=value,
        )


def test_compare_profit_scenarios_ranks_by_profit(economic_trace, economic_profile):
    target = ProfitTarget(latency_slo=queue_slo())
    result = compare_profit_scenarios(
        [
            ProfitScenario(
                "expensive", economic_trace, economic_profile, pricing(2.0), target
            ),
            ProfitScenario(
                "cheap", economic_trace, economic_profile, pricing(1.0), target
            ),
        ],
        options=options(),
    )

    assert list(result["scenario"]) == ["cheap", "expensive"]


def test_compare_profit_scenarios_rejects_mixed_currencies(
    economic_trace, economic_profile
):
    target = ProfitTarget(latency_slo=queue_slo())

    with pytest.raises(ValueError, match="same currency"):
        compare_profit_scenarios(
            [
                ProfitScenario(
                    "usd",
                    economic_trace,
                    economic_profile,
                    pricing(currency="USD"),
                    target,
                ),
                ProfitScenario(
                    "eur",
                    economic_trace,
                    economic_profile,
                    pricing(currency="EUR"),
                    target,
                ),
            ],
            options=options(),
        )
