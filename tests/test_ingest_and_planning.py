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
        "class_name",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "thinking_tokens",
        "max_output_tokens",
        "observed_latency_s",
    ]
    assert trace.frame["arrival_s"].iloc[0] == 0.0


def test_latency_plan_returns_positive_units() -> None:
    trace = slz.make_synthetic_trace(seed=123)
    profile = slz.vertex_profile("gemini-2.0-flash-001")
    result = slz.plan_capacity(
        trace,
        profile,
        slz.LatencyTarget(slz.LatencySLO(threshold_s=1.8, percentile=0.95)),
    )
    assert result.recommended_units >= 1
    assert "p95_latency_s" in result.metrics


def test_throughput_plan_returns_positive_units() -> None:
    trace = slz.make_synthetic_trace(seed=456)
    profile = slz.vertex_profile("gemini-2.0-flash-001")
    result = slz.plan_capacity(
        trace,
        profile,
        slz.ThroughputTarget(percentile=0.99, max_overload_probability=0.02),
    )
    assert result.recommended_units >= 1
    assert "worst_window_overload_probability" in result.metrics
