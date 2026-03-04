"""Tests for capacity planning."""

import numpy as np
import pandas as pd
import pytest

from slosizer.ingest import from_dataframe
from slosizer.planning import compare_scenarios, plan_capacity
from slosizer.schema import (
    CapacityProfile,
    LatencySLO,
    LatencyTarget,
    PlanOptions,
    RequestSchema,
    ThroughputTarget,
)


@pytest.fixture
def simple_profile():
    return CapacityProfile(
        provider="test",
        model="test-model",
        unit_name="GSU",
        throughput_per_unit=1000.0,
        purchase_increment=1,
        min_units=1,
        input_weight=1.0,
        cached_input_weight=0.0,
        output_weight=4.0,
        thinking_weight=4.0,
    )


@pytest.fixture
def simple_trace():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame(
        {
            "ts": np.sort(np.random.uniform(0, 3600, n)),
            "input_tokens": np.random.randint(100, 500, n),
            "cached_input_tokens": [0] * n,
            "output_tokens": np.random.randint(50, 200, n),
            "thinking_tokens": [0] * n,
        }
    )
    return from_dataframe(
        df,
        schema=RequestSchema(
            time_col="ts",
            input_tokens_col="input_tokens",
            output_tokens_col="output_tokens",
            cached_input_tokens_col="cached_input_tokens",
            thinking_tokens_col="thinking_tokens",
            class_col=None,
        ),
    )


class TestPlanCapacity:
    def test_throughput_planning(self, simple_profile, simple_trace):
        target = ThroughputTarget(percentile=0.99)
        result = plan_capacity(simple_trace, simple_profile, target)

        assert result.objective == "throughput"
        assert result.recommended_units >= simple_profile.min_units
        assert result.unit_name == "GSU"
        assert "planning_percentile" in result.metrics

    def test_throughput_with_overload_constraint(self, simple_profile, simple_trace):
        target = ThroughputTarget(
            percentile=0.99,
            max_overload_probability=0.01,
        )
        result = plan_capacity(simple_trace, simple_profile, target)

        assert result.metrics["worst_window_overload_probability"] <= 0.01

    def test_latency_planning(self, simple_profile, simple_trace):
        target = LatencyTarget(slo=LatencySLO(threshold_s=5.0, percentile=0.99))
        result = plan_capacity(simple_trace, simple_profile, target)

        assert result.objective == "latency"
        assert result.recommended_units >= simple_profile.min_units
        assert result.latency_summary is not None
        assert result.request_level is not None

    def test_headroom_factor(self, simple_profile, simple_trace):
        target = ThroughputTarget(percentile=0.99)

        result_no_headroom = plan_capacity(
            simple_trace,
            simple_profile,
            target,
            options=PlanOptions(headroom_factor=0.0),
        )
        result_with_headroom = plan_capacity(
            simple_trace,
            simple_profile,
            target,
            options=PlanOptions(headroom_factor=0.2),
        )

        assert result_with_headroom.recommended_units >= result_no_headroom.recommended_units

    def test_max_output_tokens_source(self, simple_profile, simple_trace):
        target = ThroughputTarget(percentile=0.99)

        result_observed = plan_capacity(
            simple_trace,
            simple_profile,
            target,
            options=PlanOptions(output_token_source="observed"),
        )
        result_max = plan_capacity(
            simple_trace,
            simple_profile,
            target,
            options=PlanOptions(output_token_source="max_output_tokens"),
        )

        # max_output_tokens should require equal or more capacity
        assert result_max.recommended_units >= result_observed.recommended_units

    def test_purchase_increment_respected(self, simple_trace):
        profile = CapacityProfile(
            provider="test",
            model="test-model",
            unit_name="GSU",
            throughput_per_unit=1000.0,
            purchase_increment=5,
            min_units=5,
        )
        target = ThroughputTarget(percentile=0.99)
        result = plan_capacity(simple_trace, profile, target)

        assert result.recommended_units % 5 == 0
        assert result.recommended_units >= 5

    def test_throughput_not_set_raises(self, simple_trace):
        profile = CapacityProfile(
            provider="test",
            model="test-model",
            unit_name="GSU",
            throughput_per_unit=None,
        )
        target = ThroughputTarget()
        with pytest.raises(ValueError, match="throughput_per_unit must be set"):
            plan_capacity(simple_trace, profile, target)

    def test_invalid_target_type_raises(self, simple_profile, simple_trace):
        with pytest.raises(TypeError, match="must be a ThroughputTarget or LatencyTarget"):
            plan_capacity(simple_trace, simple_profile, "invalid")

    def test_infeasible_latency_raises(self, simple_profile, simple_trace):
        # Very tight latency requirement
        target = LatencyTarget(slo=LatencySLO(threshold_s=0.0001, percentile=0.99))
        with pytest.raises(RuntimeError, match="No feasible latency plan"):
            plan_capacity(
                simple_trace,
                simple_profile,
                target,
                options=PlanOptions(max_units_to_search=10),
            )


class TestCompareScenarios:
    def test_basic_comparison(self, simple_profile, simple_trace):
        scenarios = {"baseline": simple_trace}
        targets = [
            ThroughputTarget(percentile=0.95),
            ThroughputTarget(percentile=0.99),
        ]

        result = compare_scenarios(scenarios, simple_profile, targets)

        assert len(result) == 2
        assert "scenario" in result.columns
        assert "recommended_units" in result.columns
        assert set(result["scenario"]) == {"baseline"}

    def test_multiple_scenarios(self, simple_profile, simple_trace):
        # Create a second trace with different load
        np.random.seed(123)
        n = 500
        df = pd.DataFrame(
            {
                "ts": np.sort(np.random.uniform(0, 3600, n)),
                "input_tokens": np.random.randint(100, 500, n),
                "cached_input_tokens": [0] * n,
                "output_tokens": np.random.randint(50, 200, n),
                "thinking_tokens": [0] * n,
            }
        )
        light_trace = from_dataframe(
            df,
            schema=RequestSchema(
                time_col="ts",
                input_tokens_col="input_tokens",
                output_tokens_col="output_tokens",
                cached_input_tokens_col="cached_input_tokens",
                thinking_tokens_col="thinking_tokens",
                class_col=None,
            ),
        )

        scenarios = {
            "normal": simple_trace,
            "light": light_trace,
        }
        targets = [ThroughputTarget(percentile=0.99)]

        result = compare_scenarios(scenarios, simple_profile, targets)

        assert len(result) == 2
        assert set(result["scenario"]) == {"normal", "light"}

    def test_mixed_targets(self, simple_profile, simple_trace):
        scenarios = {"baseline": simple_trace}
        targets = [
            ThroughputTarget(percentile=0.99),
            LatencyTarget(slo=LatencySLO(threshold_s=5.0, percentile=0.99)),
        ]

        result = compare_scenarios(scenarios, simple_profile, targets)

        assert len(result) == 2
        assert set(result["objective"]) == {"throughput", "latency"}
