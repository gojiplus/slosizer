"""Tests for capacity simulation."""

import numpy as np
import pandas as pd
import pytest

from slosizer.ingest import from_dataframe
from slosizer.schema import (
    BaselineLatencyModel,
    CapacityProfile,
    PlanOptions,
    RequestSchema,
)
from slosizer.simulation import (
    bucket_required_units,
    fit_baseline_latency_model,
    simulate_capacity,
    summarize_slack,
)


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
            "ts": np.arange(0, 10, 1.0),
            "input_tokens": [100] * 10,
            "cached_input_tokens": [0] * 10,
            "output_tokens": [50] * 10,
            "thinking_tokens": [0] * 10,
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


class TestBucketRequiredUnits:
    def test_basic_bucketing(self, simple_profile, simple_trace):
        result = bucket_required_units(
            simple_trace.frame,
            simple_profile,
            units=10,
            windows_s=[1.0],
            output_token_source="observed",
        )

        assert "window_s" in result.columns
        assert "required_units" in result.columns
        assert "spare_units" in result.columns
        assert "overflow_units" in result.columns
        # 10 requests at times 0-9 creates 9 buckets (0-1, 1-2, ..., 8-9)
        assert len(result) == 9

    def test_multiple_windows(self, simple_profile, simple_trace):
        result = bucket_required_units(
            simple_trace.frame,
            simple_profile,
            units=10,
            windows_s=[1.0, 5.0],
            output_token_source="observed",
        )

        window_counts = result.groupby("window_s").size()
        assert window_counts[1.0] == 9  # 0-9 seconds = 9 buckets
        assert window_counts[5.0] == 2  # 0-5, 5-10 = 2 buckets

    def test_spare_and_overflow(self, simple_profile, simple_trace):
        result = bucket_required_units(
            simple_trace.frame,
            simple_profile,
            units=0,  # No capacity
            windows_s=[1.0],
            output_token_source="observed",
        )

        # With 0 units, all required is overflow
        assert (result["spare_units"] == 0).all()
        assert (result["overflow_units"] == result["required_units"]).all()

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
        result = bucket_required_units(
            empty_frame,
            simple_profile,
            units=10,
            windows_s=[1.0],
            output_token_source="observed",
        )

        assert len(result) == 0

    def test_throughput_not_set_raises(self, simple_trace):
        profile = CapacityProfile(
            provider="test",
            model="test-model",
            unit_name="GSU",
            throughput_per_unit=None,
        )
        with pytest.raises(ValueError, match="throughput_per_unit must be set"):
            bucket_required_units(
                simple_trace.frame,
                profile,
                units=10,
                windows_s=[1.0],
                output_token_source="observed",
            )


class TestSummarizeSlack:
    def test_basic_summary(self, simple_profile, simple_trace):
        slack_table = bucket_required_units(
            simple_trace.frame,
            simple_profile,
            units=10,
            windows_s=[1.0, 5.0],
            output_token_source="observed",
        )
        summary = summarize_slack(slack_table)

        assert len(summary) == 2  # Two window sizes
        assert "avg_required_units" in summary.columns
        assert "overload_probability" in summary.columns
        assert "avg_spare_fraction" in summary.columns

    def test_empty_table(self):
        result = summarize_slack(pd.DataFrame())
        assert len(result) == 0


class TestFitBaselineLatencyModel:
    def test_insufficient_data_returns_default(self):
        df = pd.DataFrame(
            {
                "ts": [0.0, 1.0],
                "input_tokens": [100, 200],
                "cached_input_tokens": [0, 0],
                "output_tokens": [50, 100],
                "thinking_tokens": [0, 0],
                "latency_s": [0.5, 1.0],
            }
        )
        trace = from_dataframe(
            df,
            schema=RequestSchema(),
        )
        model = fit_baseline_latency_model(trace)

        # Should return default model due to < 20 samples
        assert model == BaselineLatencyModel()

    def test_no_latency_data_returns_default(self):
        df = pd.DataFrame(
            {
                "ts": np.arange(50),
                "input_tokens": [100] * 50,
                "cached_input_tokens": [0] * 50,
                "output_tokens": [50] * 50,
                "thinking_tokens": [0] * 50,
            }
        )
        trace = from_dataframe(
            df,
            schema=RequestSchema(latency_col=None),
        )
        model = fit_baseline_latency_model(trace)

        assert model == BaselineLatencyModel()

    def test_fits_with_sufficient_data(self):
        np.random.seed(42)
        n = 100
        input_tokens = np.random.randint(100, 1000, n)
        output_tokens = np.random.randint(50, 200, n)
        # Synthetic latency = 0.1 + 0.001*input + 0.01*output + noise
        latency = 0.1 + 0.001 * input_tokens + 0.01 * output_tokens + np.random.normal(0, 0.05, n)

        df = pd.DataFrame(
            {
                "ts": np.arange(n).astype(float),
                "input_tokens": input_tokens,
                "cached_input_tokens": [0] * n,
                "output_tokens": output_tokens,
                "thinking_tokens": [0] * n,
                "latency_s": latency,
            }
        )
        trace = from_dataframe(df, schema=RequestSchema())
        model = fit_baseline_latency_model(trace)

        # Should have non-default coefficients
        assert model != BaselineLatencyModel()
        # Coefficients should be reasonable
        assert model.intercept_s >= 0
        assert model.input_token_s >= 0
        assert model.output_token_s >= 0

    def test_ignores_nan_latency_rows(self):
        from slosizer.schema import RequestTrace

        n = 50
        frame = pd.DataFrame(
            {
                "arrival_s": np.arange(n).astype(float),
                "class_name": ["default"] * n,
                "input_tokens": [100] * n,
                "cached_input_tokens": [0] * n,
                "output_tokens": [50] * n,
                "thinking_tokens": [0] * n,
                "max_output_tokens": [200] * n,
                "observed_latency_s": [0.5] * 30 + [float("nan")] * 20,
            }
        )
        trace = RequestTrace(frame=frame, schema=RequestSchema())
        model = fit_baseline_latency_model(trace)

        # Should fit on the 30 valid rows
        assert model != BaselineLatencyModel()


class TestSimulateCapacity:
    def test_basic_simulation(self, simple_profile, simple_trace):
        result = simulate_capacity(
            simple_trace,
            simple_profile,
            units=10,
        )

        assert result.units == 10
        assert result.unit_name == "GSU"
        assert "queue_delay_s" in result.request_level.columns
        assert "total_latency_s" in result.request_level.columns
        assert len(result.latency_summary) == 1
        assert len(result.slack_summary) == 3  # Default 3 windows

    def test_custom_windows(self, simple_profile, simple_trace):
        result = simulate_capacity(
            simple_trace,
            simple_profile,
            units=10,
            windows_s=(2.0, 10.0),
        )

        assert len(result.slack_summary) == 2
        assert set(result.slack_summary["window_s"]) == {2.0, 10.0}

    def test_queue_delay_increases_with_load(self, simple_profile):
        # High load trace
        df = pd.DataFrame(
            {
                "ts": np.arange(0, 1, 0.01),  # 100 requests in 1 second
                "input_tokens": [1000] * 100,
                "cached_input_tokens": [0] * 100,
                "output_tokens": [500] * 100,
                "thinking_tokens": [0] * 100,
            }
        )
        trace = from_dataframe(
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

        result_low = simulate_capacity(trace, simple_profile, units=100)
        result_high = simulate_capacity(trace, simple_profile, units=1)

        # More capacity = less queue delay
        assert (
            result_low.latency_summary["mean_queue_delay_s"].iloc[0]
            < result_high.latency_summary["mean_queue_delay_s"].iloc[0]
        )

    def test_throughput_not_set_raises(self, simple_trace):
        profile = CapacityProfile(
            provider="test",
            model="test-model",
            unit_name="GSU",
            throughput_per_unit=None,
        )
        with pytest.raises(ValueError, match="throughput_per_unit must be set"):
            simulate_capacity(simple_trace, profile, units=10)

    def test_custom_baseline_model(self, simple_profile, simple_trace):
        custom_model = BaselineLatencyModel(
            intercept_s=1.0,
            input_token_s=0.0,
            output_token_s=0.0,
        )
        options = PlanOptions(baseline_latency_model=custom_model)
        result = simulate_capacity(
            simple_trace,
            simple_profile,
            units=100,
            options=options,
        )

        # All requests should have ~1s baseline latency
        np.testing.assert_array_almost_equal(
            result.request_level["baseline_latency_s"],
            [1.0] * len(result.request_level),
            decimal=1,
        )
