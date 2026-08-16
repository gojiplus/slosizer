import pandas as pd

import slosizer as slz


def test_ingest_normalizes_columns() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [10.0, 11.5],
            "prompt_tokens": [100, 200],
            "completion_tokens": [50, 70],
        }
    )
    trace = slz.from_dataframe(
        df,
        schema=slz.RequestSchema(
            time_col="timestamp",
            input_tokens_col="prompt_tokens",
            output_tokens_col="completion_tokens",
        ),
    )
    assert list(trace.frame.columns) == [
        "arrival_s",
        "request_id",
        "class_name",
        "request_model",
        "response_model",
        "service_tier",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "thinking_tokens",
        "max_output_tokens",
        "observed_latency_s",
        "business_value",
    ]
    assert trace.frame["arrival_s"].iloc[0] == 0.0


def test_latency_plan_returns_positive_units() -> None:
    trace = slz.make_synthetic_trace(seed=123)
    profile = slz.vertex_profile("gemini-2.5-flash-lite")
    result = slz.plan_capacity(
        trace,
        profile,
        slz.LatencyTarget(slz.LatencySLO(threshold_s=1.8, percentile=0.95)),
    )
    assert result.recommended_units >= 1
    assert "p95_latency_s" in result.metrics


def test_throughput_plan_returns_positive_units() -> None:
    trace = slz.make_synthetic_trace(seed=456)
    profile = slz.vertex_profile("gemini-2.5-flash-lite")
    result = slz.plan_capacity(
        trace,
        profile,
        slz.ThroughputTarget(percentile=0.99, max_overload_probability=0.02),
    )
    assert result.recommended_units >= 1
    assert "worst_window_overload_probability" in result.metrics


def test_optimize_trace_preserves_request_and_business_metadata() -> None:
    frame = pd.DataFrame(
        {
            "ts": [0.0, 1.0],
            "request_id": ["req-1", "req-2"],
            "class_name": ["chat", "rag"],
            "request_model": ["requested-a", "requested-b"],
            "response_model": ["served-a", "served-b"],
            "service_tier": ["standard", "priority"],
            "input_tokens": [100, 200],
            "output_tokens": [50, 75],
            "latency_s": [1.0, 1.5],
            "business_value": [0.25, 0.75],
        }
    )
    trace = slz.from_dataframe(frame, schema=slz.RequestSchema())

    optimized = slz.optimize_trace(trace)

    for column in (
        "request_id",
        "request_model",
        "response_model",
        "service_tier",
        "observed_latency_s",
        "business_value",
    ):
        pd.testing.assert_series_equal(
            optimized.frame[column], trace.frame[column], check_names=False
        )


def test_optimized_synthetic_trace_preserves_missing_optional_latency() -> None:
    trace = slz.make_synthetic_trace(seed=42, scenario="optimized")

    assert trace.frame["observed_latency_s"].isna().all()
