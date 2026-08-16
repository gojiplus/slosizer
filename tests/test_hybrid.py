"""Tests for hybrid capacity planning."""

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from slosizer import (
    HybridPlanResult,
    HybridTarget,
    LatencySLO,
    PaygoPricing,
    PlanOptions,
    ProvisionedPricing,
    RateCard,
    RequestSchema,
    from_dataframe,
    plan_hybrid_capacity,
)
from slosizer.hybrid import _hybrid_candidates, compute_overflow_tokens
from slosizer.schema import CapacityProfile
from slosizer.simulation import bucket_with_tokens


@pytest.fixture
def simple_profile():
    return CapacityProfile(
        provider="test",
        model="test-model",
        unit_name="GSU",
        throughput_per_unit=1000.0,
        input_weight=1.0,
        cached_input_weight=0.0,
        output_weight=4.0,
        thinking_weight=4.0,
    )


@pytest.fixture
def simple_trace():
    df = pd.DataFrame(
        {
            "ts": np.arange(0, 100, 1.0),
            "input_tokens": [100] * 100,
            "cached_input_tokens": [0] * 100,
            "output_tokens": [50] * 100,
            "thinking_tokens": [0] * 100,
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


@pytest.fixture
def bursty_trace():
    times = list(np.arange(0, 50, 1.0)) + list(np.arange(50, 51, 0.02))
    n = len(times)
    df = pd.DataFrame(
        {
            "ts": times,
            "input_tokens": [100] * n,
            "cached_input_tokens": [0] * n,
            "output_tokens": [50] * n,
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


@pytest.fixture
def standard_pricing():
    return RateCard(
        provisioned=ProvisionedPricing(cost_per_unit_hour=2.50),
        paygo=PaygoPricing(
            input_cost_per_million=0.30,
            output_cost_per_million=2.50,
        ),
    )


class TestPaygoPricing:
    def test_valid_pricing(self):
        pricing = PaygoPricing(
            input_cost_per_million=0.30, output_cost_per_million=2.50
        )
        assert pricing.input_cost_per_million == 0.30
        assert pricing.output_cost_per_million == 2.50

    def test_zero_pricing_allowed(self):
        pricing = PaygoPricing(input_cost_per_million=0.0, output_cost_per_million=0.0)
        assert pricing.input_cost_per_million == 0.0
        assert pricing.output_cost_per_million == 0.0

    def test_negative_input_cost_raises(self):
        with pytest.raises(
            ValueError, match="input_cost_per_million must be non-negative"
        ):
            PaygoPricing(input_cost_per_million=-0.30, output_cost_per_million=2.50)

    def test_negative_output_cost_raises(self):
        with pytest.raises(
            ValueError, match="output_cost_per_million must be non-negative"
        ):
            PaygoPricing(input_cost_per_million=0.30, output_cost_per_million=-2.50)

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_non_finite_cost_raises(self, value):
        with pytest.raises(ValueError, match="must be finite"):
            PaygoPricing(
                input_cost_per_million=value,
                output_cost_per_million=2.50,
            )

    def test_optional_token_prices(self):
        pricing = PaygoPricing(
            input_cost_per_million=1.0,
            cached_input_cost_per_million=0.1,
            output_cost_per_million=4.0,
            thinking_cost_per_million=5.0,
        )

        assert pricing.effective_cached_input_cost_per_million == 0.1
        assert pricing.effective_thinking_cost_per_million == 5.0

    def test_optional_token_prices_fall_back_conservatively(self):
        pricing = PaygoPricing(
            input_cost_per_million=1.0,
            output_cost_per_million=4.0,
        )

        assert pricing.effective_cached_input_cost_per_million == 1.0
        assert pricing.effective_thinking_cost_per_million == 4.0


class TestProvisionedPricing:
    def test_valid_pricing(self):
        pricing = ProvisionedPricing(cost_per_unit_hour=2.50)
        assert pricing.cost_per_unit_hour == 2.50

    def test_zero_pricing_allowed(self):
        pricing = ProvisionedPricing(cost_per_unit_hour=0.0)
        assert pricing.cost_per_unit_hour == 0.0

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError, match="cost_per_unit_hour must be non-negative"):
            ProvisionedPricing(cost_per_unit_hour=-2.50)

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_non_finite_cost_raises(self, value):
        with pytest.raises(ValueError, match="cost_per_unit_hour must be finite"):
            ProvisionedPricing(cost_per_unit_hour=value)


class TestRateCard:
    def test_currency_is_normalized(self):
        rate_card = RateCard(
            provisioned=ProvisionedPricing(cost_per_unit_hour=1.0),
            currency="usd",
        )

        assert rate_card.currency == "USD"

    def test_effective_date_order_is_validated(self):
        with pytest.raises(ValueError, match="effective_to"):
            RateCard(
                provisioned=ProvisionedPricing(cost_per_unit_hour=1.0),
                effective_from=date(2026, 8, 15),
                effective_to=date(2026, 8, 14),
            )


class TestHybridTarget:
    def test_cost_optimal_strategy(self):
        target = HybridTarget(strategy="cost_optimal")
        assert target.strategy == "cost_optimal"
        assert target.provision_percentile is None
        assert target.latency_slo is None

    def test_percentile_split_strategy(self):
        target = HybridTarget(strategy="percentile_split", provision_percentile=0.50)
        assert target.strategy == "percentile_split"
        assert target.provision_percentile == 0.50

    def test_percentile_split_without_percentile_raises(self):
        with pytest.raises(ValueError, match="provision_percentile is required"):
            HybridTarget(strategy="percentile_split")

    def test_invalid_percentile_raises(self):
        with pytest.raises(
            ValueError, match="provision_percentile must be in \\(0, 1\\)"
        ):
            HybridTarget(strategy="percentile_split", provision_percentile=1.5)

        with pytest.raises(
            ValueError, match="provision_percentile must be in \\(0, 1\\)"
        ):
            HybridTarget(strategy="percentile_split", provision_percentile=0.0)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="unknown hybrid strategy"):
            HybridTarget(strategy="unknown")

    def test_with_latency_slo(self):
        slo = LatencySLO(threshold_s=1.5, percentile=0.99)
        target = HybridTarget(strategy="cost_optimal", latency_slo=slo)
        assert target.latency_slo is not None
        assert target.latency_slo.threshold_s == 1.5

    def test_label_cost_optimal(self):
        target = HybridTarget(strategy="cost_optimal")
        assert target.label() == "hybrid-cost-optimal"

    def test_label_percentile_split(self):
        target = HybridTarget(strategy="percentile_split", provision_percentile=0.50)
        assert target.label() == "hybrid-p50-split"

    def test_label_with_slo(self):
        slo = LatencySLO(threshold_s=1.5, percentile=0.99)
        target = HybridTarget(strategy="cost_optimal", latency_slo=slo)
        assert "slo-p99<=1.5s" in target.label()


class TestBucketWithTokens:
    def test_basic_bucketing(self, simple_profile, simple_trace):
        result = bucket_with_tokens(
            simple_trace.frame,
            simple_profile,
            units=10,
            window_s=1.0,
            output_token_source="observed",
        )

        assert "bucket_start_s" in result.columns
        assert "required_units" in result.columns
        assert "overflow_units" in result.columns
        assert "overflow_fraction" in result.columns
        assert "input_tokens" in result.columns
        assert "output_tokens" in result.columns

    def test_token_counts_aggregated(self, simple_profile, simple_trace):
        result = bucket_with_tokens(
            simple_trace.frame,
            simple_profile,
            units=10,
            window_s=1.0,
            output_token_source="observed",
        )

        total_input = result["input_tokens"].sum()
        assert total_input == 100 * 100

    def test_empty_intermediate_buckets_do_not_divide_by_zero(
        self, simple_profile, simple_trace
    ):
        sparse_frame = simple_trace.frame.iloc[[0, -1]]

        with np.errstate(divide="raise", invalid="raise"):
            result = bucket_with_tokens(
                sparse_frame,
                simple_profile,
                units=1,
                window_s=1.0,
                output_token_source="observed",
            )

        assert len(result) == 100
        assert result["input_tokens"].sum() == 200
        assert (
            result.loc[result["required_units"] == 0, "overflow_fraction"].eq(0).all()
        )

    def test_overflow_fraction_calculation(self, simple_profile, simple_trace):
        result = bucket_with_tokens(
            simple_trace.frame,
            simple_profile,
            units=0,
            window_s=1.0,
            output_token_source="observed",
        )

        assert (result["overflow_fraction"] == 1.0).all()

    def test_no_overflow_with_high_capacity(self, simple_profile, simple_trace):
        result = bucket_with_tokens(
            simple_trace.frame,
            simple_profile,
            units=100,
            window_s=1.0,
            output_token_source="observed",
        )

        assert (result["overflow_fraction"] == 0.0).all()

    def test_empty_frame(self, simple_profile):
        empty_frame = pd.DataFrame(
            {
                "arrival_s": [],
                "input_tokens": [],
                "cached_input_tokens": [],
                "output_tokens": [],
                "thinking_tokens": [],
            }
        )
        result = bucket_with_tokens(
            empty_frame,
            simple_profile,
            units=10,
            window_s=1.0,
            output_token_source="observed",
        )

        assert len(result) == 0


class TestComputeOverflowTokens:
    def test_no_overflow_with_high_capacity(self, simple_profile, simple_trace):
        result = compute_overflow_tokens(
            simple_trace,
            simple_profile,
            units=100,
            window_s=1.0,
            output_token_source="observed",
        )

        assert result["overflow_input_tokens"] == 0.0
        assert result["overflow_output_tokens"] == 0.0
        assert result["overflow_fraction"] == 0.0

    def test_full_overflow_with_zero_capacity(self, simple_profile, simple_trace):
        result = compute_overflow_tokens(
            simple_trace,
            simple_profile,
            units=0,
            window_s=1.0,
            output_token_source="observed",
        )

        assert result["overflow_input_tokens"] > 0
        assert result["overflow_output_tokens"] > 0
        assert result["overflow_fraction"] == 1.0

    def test_trace_duration_calculation(self, simple_profile, simple_trace):
        result = compute_overflow_tokens(
            simple_trace,
            simple_profile,
            units=10,
            window_s=1.0,
            output_token_source="observed",
        )

        assert result["trace_duration_hours"] == pytest.approx(99.0 / 3600.0)


class TestPlanHybridCapacity:
    def test_candidates_share_provider_purchase_grid(self):
        profile = CapacityProfile(
            provider="test",
            model="test-model",
            unit_name="GSU",
            throughput_per_unit=1000.0,
            min_units=3,
            purchase_increment=5,
        )

        assert _hybrid_candidates(profile, 16) == [0, 5, 10, 15]

    def test_cost_optimal_basic(self, simple_profile, simple_trace, standard_pricing):
        target = HybridTarget(strategy="cost_optimal")
        result = plan_hybrid_capacity(
            simple_trace, simple_profile, standard_pricing, target
        )

        assert isinstance(result, HybridPlanResult)
        assert result.provisioned_units >= 0
        assert result.total_cost_hourly >= 0
        assert result.total_cost_hourly == pytest.approx(
            result.provisioned_cost_hourly + result.paygo_cost_hourly
        )

    def test_percentile_split_basic(
        self, simple_profile, simple_trace, standard_pricing
    ):
        target = HybridTarget(strategy="percentile_split", provision_percentile=0.50)
        result = plan_hybrid_capacity(
            simple_trace, simple_profile, standard_pricing, target
        )

        assert isinstance(result, HybridPlanResult)
        assert result.provisioned_units >= 0

    def test_percentile_split_respects_search_limit(
        self, simple_profile, simple_trace, standard_pricing
    ):
        profile = replace(simple_profile, throughput_per_unit=100.0)

        with pytest.raises(RuntimeError, match="above the search limit"):
            plan_hybrid_capacity(
                simple_trace,
                profile,
                standard_pricing,
                HybridTarget(strategy="percentile_split", provision_percentile=0.50),
                options=PlanOptions(max_units_to_search=1),
            )

    def test_hybrid_cost_le_full_provision(
        self, simple_profile, bursty_trace, standard_pricing
    ):
        target = HybridTarget(strategy="cost_optimal")
        result = plan_hybrid_capacity(
            bursty_trace, simple_profile, standard_pricing, target
        )

        assert result.total_cost_hourly <= result.full_provision_cost_hourly + 0.01

    def test_savings_calculation(self, simple_profile, bursty_trace, standard_pricing):
        target = HybridTarget(strategy="cost_optimal")
        result = plan_hybrid_capacity(
            bursty_trace, simple_profile, standard_pricing, target
        )

        expected_savings = result.full_provision_cost_hourly - result.total_cost_hourly
        assert result.savings_vs_full_provision == pytest.approx(expected_savings)

        if result.full_provision_cost_hourly > 0:
            expected_pct = (expected_savings / result.full_provision_cost_hourly) * 100
            assert result.savings_percent == pytest.approx(expected_pct)

    def test_with_latency_slo(self, simple_profile, simple_trace, standard_pricing):
        slo = LatencySLO(threshold_s=2.0, percentile=0.99)
        target = HybridTarget(strategy="cost_optimal", latency_slo=slo)
        result = plan_hybrid_capacity(
            simple_trace, simple_profile, standard_pricing, target
        )

        assert result.provisioned_units > 0

    def test_nonzero_candidates_respect_provider_minimum(
        self, simple_profile, simple_trace
    ):
        profile = replace(simple_profile, min_units=15, purchase_increment=5)
        expensive_paygo = RateCard(
            provisioned=ProvisionedPricing(cost_per_unit_hour=0.01),
            paygo=PaygoPricing(
                input_cost_per_million=10000.0,
                output_cost_per_million=10000.0,
            ),
        )
        result = plan_hybrid_capacity(
            simple_trace,
            profile,
            expensive_paygo,
            HybridTarget(strategy="cost_optimal"),
        )

        assert result.provisioned_units >= 15
        assert (result.provisioned_units - 15) % 5 == 0

    def test_pricing_scope_must_match_profile(
        self, simple_profile, simple_trace, standard_pricing
    ):
        scoped_pricing = replace(standard_pricing, model="different-model")
        with pytest.raises(ValueError, match="does not match"):
            plan_hybrid_capacity(
                simple_trace,
                simple_profile,
                scoped_pricing,
                HybridTarget(strategy="cost_optimal"),
            )

    def test_paygo_prices_are_required(self, simple_profile, simple_trace):
        rate_card = RateCard(provisioned=ProvisionedPricing(cost_per_unit_hour=1.0))
        with pytest.raises(ValueError, match=r"pricing\.paygo is required"):
            plan_hybrid_capacity(
                simple_trace,
                simple_profile,
                rate_card,
                HybridTarget(strategy="cost_optimal"),
            )

    def test_full_paygo_when_cheaper(self, simple_profile, simple_trace):
        expensive_provisioned = RateCard(
            provisioned=ProvisionedPricing(cost_per_unit_hour=1000.0),
            paygo=PaygoPricing(
                input_cost_per_million=0.001,
                output_cost_per_million=0.001,
            ),
        )
        target = HybridTarget(strategy="cost_optimal")
        result = plan_hybrid_capacity(
            simple_trace, simple_profile, expensive_provisioned, target
        )

        assert result.provisioned_units == 0
        assert result.paygo_cost_hourly > 0

    def test_full_provisioned_when_cheaper(self, simple_profile, simple_trace):
        expensive_paygo = RateCard(
            provisioned=ProvisionedPricing(cost_per_unit_hour=0.01),
            paygo=PaygoPricing(
                input_cost_per_million=10000.0,
                output_cost_per_million=10000.0,
            ),
        )
        target = HybridTarget(strategy="cost_optimal")
        result = plan_hybrid_capacity(
            simple_trace, simple_profile, expensive_paygo, target
        )

        assert result.provisioned_units > 0
        assert result.paygo_cost_hourly == pytest.approx(0.0, abs=0.01)

    def test_headroom_factor_applied(
        self, simple_profile, simple_trace, standard_pricing
    ):
        target = HybridTarget(strategy="percentile_split", provision_percentile=0.50)
        options_no_headroom = PlanOptions(headroom_factor=0.0)
        options_with_headroom = PlanOptions(headroom_factor=0.20)

        result_no = plan_hybrid_capacity(
            simple_trace,
            simple_profile,
            standard_pricing,
            target,
            options=options_no_headroom,
        )
        result_with = plan_hybrid_capacity(
            simple_trace,
            simple_profile,
            standard_pricing,
            target,
            options=options_with_headroom,
        )

        assert result_with.provisioned_units >= result_no.provisioned_units

    def test_cost_optimal_rejects_post_optimization_headroom(
        self, simple_profile, simple_trace, standard_pricing
    ):
        with pytest.raises(ValueError, match="headroom_factor"):
            plan_hybrid_capacity(
                simple_trace,
                simple_profile,
                standard_pricing,
                HybridTarget(strategy="cost_optimal"),
                options=PlanOptions(headroom_factor=0.2),
            )

    def test_throughput_not_set_raises(self, simple_trace, standard_pricing):
        profile = CapacityProfile(
            provider="test",
            model="test-model",
            unit_name="GSU",
            throughput_per_unit=None,
        )
        target = HybridTarget(strategy="cost_optimal")
        with pytest.raises(ValueError, match="throughput_per_unit must be set"):
            plan_hybrid_capacity(simple_trace, profile, standard_pricing, target)

    def test_result_has_slack_summary(
        self, simple_profile, simple_trace, standard_pricing
    ):
        target = HybridTarget(strategy="cost_optimal")
        result = plan_hybrid_capacity(
            simple_trace, simple_profile, standard_pricing, target
        )

        assert result.slack_summary is not None
        assert len(result.slack_summary) > 0

    def test_result_as_dict(self, simple_profile, simple_trace, standard_pricing):
        target = HybridTarget(strategy="cost_optimal")
        result = plan_hybrid_capacity(
            simple_trace, simple_profile, standard_pricing, target
        )

        result_dict = result.as_dict()
        assert "provisioned_units" in result_dict
        assert "total_cost_hourly" in result_dict
        assert "savings_percent" in result_dict

    def test_assumptions_populated(
        self, simple_profile, simple_trace, standard_pricing
    ):
        target = HybridTarget(strategy="cost_optimal")
        result = plan_hybrid_capacity(
            simple_trace, simple_profile, standard_pricing, target
        )

        assert "strategy" in result.assumptions
        assert "provisioned_cost_per_unit_hour" in result.assumptions
        assert result.assumptions["strategy"] == "cost_optimal"
