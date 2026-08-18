"""Capacity simulation for queue-based latency modeling.

This module simulates request processing with finite capacity to estimate
latency distributions and capacity utilization.
"""

import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd

from slosizer._utils import adjusted_work, empirical_quantile
from slosizer.schema import (
    BaselineLatencyModel,
    CapacityProfile,
    PlanOptions,
    RequestTrace,
    SimulationResult,
)


def _bucket_edges(max_time: float, window_s: float) -> np.ndarray:
    """Return fixed-width edges including the bucket containing ``max_time``."""
    n_buckets = int(np.floor(max_time / window_s)) + 1
    return np.arange(n_buckets + 1, dtype=float) * window_s


def fit_baseline_latency_model(trace: RequestTrace) -> BaselineLatencyModel:
    """Fit a linear latency model from observed latencies.

    Uses ordinary least squares to fit latency as a function of token counts.
    Coefficients are constrained to be non-negative. Only rows with valid
    (non-NaN) latency values are used for fitting.

    Args:
        trace: Request trace with observed latencies.

    Returns:
        Fitted baseline latency model, or default model if insufficient data.
    """
    frame = trace.frame
    observed = frame["observed_latency_s"]
    valid_mask = observed.notna()
    n_valid = int(valid_mask.sum())

    if n_valid < 20:
        warnings.warn(
            f"Insufficient latency data ({n_valid} samples, need 20+). "
            "Using default latency model. "
            "Provide observed_latency_s values for more accurate estimates.",
            UserWarning,
            stacklevel=2,
        )
        return BaselineLatencyModel()

    valid_frame = frame.loc[valid_mask]
    design = np.column_stack(
        [
            np.ones(n_valid),
            valid_frame["input_tokens"].to_numpy(dtype=float),
            valid_frame["cached_input_tokens"].to_numpy(dtype=float),
            valid_frame["output_tokens"].to_numpy(dtype=float),
            valid_frame["thinking_tokens"].to_numpy(dtype=float),
        ]
    )
    target = observed.loc[valid_mask].to_numpy(dtype=float)
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    coeffs = np.maximum(coeffs, 0.0)
    return BaselineLatencyModel(
        intercept_s=float(coeffs[0]),
        input_token_s=float(coeffs[1]),
        cached_input_token_s=float(coeffs[2]),
        output_token_s=float(coeffs[3]),
        thinking_token_s=float(coeffs[4]),
    )


def bucket_required_units(
    frame: pd.DataFrame,
    profile: CapacityProfile,
    *,
    units: int,
    windows_s: Iterable[float],
    output_token_source: str,
) -> pd.DataFrame:
    """Compute required capacity units per time bucket.

    Divides the trace into fixed-width time windows and calculates the
    capacity units needed to serve all requests in each window.

    Args:
        frame: DataFrame with canonical columns.
        profile: Capacity profile with throughput settings.
        units: Reserved capacity units to compare against.
        windows_s: Time window sizes in seconds.
        output_token_source: Source for output tokens.

    Returns:
        DataFrame with required_units, spare_units, and overflow_units per bucket.

    Raises:
        ValueError: If profile.throughput_per_unit is not set.
    """
    if profile.throughput_per_unit is None:
        raise ValueError("profile.throughput_per_unit must be set before simulation.")

    arrivals = frame["arrival_s"].to_numpy(dtype=float)
    work = adjusted_work(frame, profile, output_token_source=output_token_source)
    if len(arrivals) == 0:
        return pd.DataFrame(
            columns=pd.Index(
                [
                    "window_s",
                    "bucket_start_s",
                    "required_units",
                    "spare_units",
                    "overflow_units",
                    "reserved_units",
                ]
            )
        )

    rows: list[dict[str, float]] = []
    max_time = float(arrivals.max())
    for window_s in windows_s:
        edges = _bucket_edges(max_time, window_s)
        bucket_index = np.digitize(arrivals, edges, right=False) - 1
        required_units = np.zeros(len(edges) - 1, dtype=float)
        scale = float(profile.throughput_per_unit) * float(window_s)
        for idx, value in zip(bucket_index, work, strict=True):
            if 0 <= idx < len(required_units):
                required_units[idx] += value / scale
        for bucket_start_s, required in zip(edges[:-1], required_units, strict=True):
            rows.append(
                {
                    "window_s": float(window_s),
                    "bucket_start_s": float(bucket_start_s),
                    "required_units": float(required),
                    "spare_units": max(0.0, float(units) - float(required)),
                    "overflow_units": max(0.0, float(required) - float(units)),
                    "reserved_units": float(units),
                }
            )
    return pd.DataFrame(rows)


def bucket_with_tokens(
    frame: pd.DataFrame,
    profile: CapacityProfile,
    *,
    units: int,
    window_s: float,
    output_token_source: str,
) -> pd.DataFrame:
    """Compute required units and token counts per time bucket.

    Similar to bucket_required_units but also tracks raw token counts per bucket
    for overflow cost calculation. Uses a single window size for efficiency.

    Args:
        frame: DataFrame with canonical columns.
        profile: Capacity profile with throughput settings.
        units: Reserved capacity units to compare against.
        window_s: Time window size in seconds.
        output_token_source: Source for output tokens.

    Returns:
        DataFrame with required_units, overflow metrics, and token counts per bucket.

    Raises:
        ValueError: If profile.throughput_per_unit is not set.
    """
    from slosizer._utils import selected_output_tokens

    if profile.throughput_per_unit is None:
        raise ValueError("profile.throughput_per_unit must be set before simulation.")

    arrivals = frame["arrival_s"].to_numpy(dtype=float)
    work = adjusted_work(frame, profile, output_token_source=output_token_source)

    input_tokens = frame["input_tokens"].to_numpy(dtype=float)
    cached_input = frame["cached_input_tokens"].to_numpy(dtype=float)
    output_tokens = selected_output_tokens(frame, output_token_source)
    thinking_tokens = frame["thinking_tokens"].to_numpy(dtype=float)

    if len(arrivals) == 0:
        return pd.DataFrame(
            columns=pd.Index(
                [
                    "bucket_start_s",
                    "window_s",
                    "required_units",
                    "overflow_units",
                    "overflow_fraction",
                    "input_tokens",
                    "cached_input_tokens",
                    "output_tokens",
                    "thinking_tokens",
                    "total_work",
                ]
            )
        )

    max_time = float(arrivals.max())
    edges = _bucket_edges(max_time, window_s)

    n_buckets = len(edges) - 1
    bucket_index = np.digitize(arrivals, edges, right=False) - 1

    bucket_work = np.zeros(n_buckets, dtype=float)
    bucket_input = np.zeros(n_buckets, dtype=float)
    bucket_cached = np.zeros(n_buckets, dtype=float)
    bucket_output = np.zeros(n_buckets, dtype=float)
    bucket_thinking = np.zeros(n_buckets, dtype=float)

    scale = float(profile.throughput_per_unit) * float(window_s)

    for i, (idx, w) in enumerate(zip(bucket_index, work, strict=True)):
        if 0 <= idx < n_buckets:
            bucket_work[idx] += w
            bucket_input[idx] += input_tokens[i]
            bucket_cached[idx] += cached_input[i]
            bucket_output[idx] += output_tokens[i]
            bucket_thinking[idx] += thinking_tokens[i]

    required_units = bucket_work / scale
    capacity = float(units)
    overflow_units = np.maximum(0.0, required_units - capacity)
    overflow_fraction = np.divide(
        overflow_units,
        required_units,
        out=np.zeros_like(required_units),
        where=required_units > 0,
    )
    overflow_fraction = np.minimum(1.0, overflow_fraction)

    return pd.DataFrame(
        {
            "bucket_start_s": edges[:-1],
            "window_s": float(window_s),
            "required_units": required_units,
            "overflow_units": overflow_units,
            "overflow_fraction": overflow_fraction,
            "input_tokens": bucket_input,
            "cached_input_tokens": bucket_cached,
            "output_tokens": bucket_output,
            "thinking_tokens": bucket_thinking,
            "total_work": bucket_work,
        }
    )


def summarize_slack(slack_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize spare capacity statistics by time window.

    Args:
        slack_table: Output from bucket_required_units.

    Returns:
        DataFrame with aggregate statistics per window size.
    """
    if slack_table.empty:
        return pd.DataFrame()
    rows = []
    grouped = slack_table.groupby("window_s", sort=True)
    for _, group in grouped:
        window_s = float(group["window_s"].iloc[0])
        reserved = float(group["reserved_units"].iloc[0])
        required = group["required_units"]
        spare = group["spare_units"]
        overflow = group["overflow_units"]
        rows.append(
            {
                "window_s": float(window_s),
                "reserved_units": reserved,
                "avg_required_units": float(required.mean()),
                "p95_required_units": float(required.quantile(0.95)),
                "p99_required_units": float(required.quantile(0.99)),
                "avg_spare_units": float(spare.mean()),
                "avg_spare_fraction": float((spare / max(reserved, 1e-9)).mean())
                if reserved
                else 0.0,
                "overload_probability": float((overflow > 0).mean()),
                "expected_overflow_units": float(overflow.mean()),
            }
        )
    return pd.DataFrame(rows)


def simulate_capacity(
    trace: RequestTrace,
    profile: CapacityProfile,
    *,
    units: int,
    options: PlanOptions | None = None,
    windows_s: tuple[float, ...] = (1.0, 5.0, 30.0),
) -> SimulationResult:
    """Simulate request processing with fixed capacity.

    Models a simple FIFO queue where requests arrive and are processed
    at a rate determined by the reserved capacity.

    Args:
        trace: Request trace to simulate.
        profile: Capacity profile with throughput settings.
        units: Number of reserved capacity units.
        options: Planning options including output token source.
        windows_s: Time window sizes for slack analysis.

    Returns:
        SimulationResult with latency and slack statistics.

    Raises:
        ValueError: If ``profile.throughput_per_unit`` is not set, ``units`` is
            less than one, or the trace has fewer than two distinct arrival
            times.
    """
    if options is None:
        options = PlanOptions()

    if profile.throughput_per_unit is None:
        raise ValueError("profile.throughput_per_unit must be set before simulation.")
    if units < 1:
        raise ValueError("units must be at least 1 for latency simulation.")

    frame = trace.frame.copy()
    arrivals = frame["arrival_s"].to_numpy(dtype=float)
    if len(arrivals) < 2:
        raise ValueError("latency simulation requires at least two requests.")
    duration_s = float(arrivals.max() - arrivals.min())
    if duration_s <= 0:
        raise ValueError("latency simulation requires a positive trace duration.")

    baseline_model = options.baseline_latency_model or fit_baseline_latency_model(trace)
    work = adjusted_work(
        frame, profile, output_token_source=options.output_token_source
    )
    baseline = baseline_model.predict(frame)
    service_rate = float(units) * float(profile.throughput_per_unit)

    backlog = 0.0
    last_t = float(arrivals[0]) if len(arrivals) else 0.0
    queue_delay = np.zeros(len(frame), dtype=float)

    for idx, arrival in enumerate(arrivals):
        elapsed = float(arrival) - last_t
        backlog = max(0.0, backlog - service_rate * elapsed)
        queue_delay[idx] = backlog / service_rate if service_rate > 0 else float("inf")
        backlog += work[idx]
        last_t = float(arrival)

    total_latency = baseline + queue_delay
    frame["adjusted_work"] = work
    frame["baseline_latency_s"] = baseline
    frame["queue_delay_s"] = queue_delay
    frame["total_latency_s"] = total_latency

    latency_summary = pd.DataFrame(
        [
            {
                "mean_latency_s": float(total_latency.mean()),
                "p50_latency_s": empirical_quantile(total_latency, 0.50),
                "p95_latency_s": empirical_quantile(total_latency, 0.95),
                "p99_latency_s": empirical_quantile(total_latency, 0.99),
                "mean_queue_delay_s": float(queue_delay.mean()),
                "p95_queue_delay_s": empirical_quantile(queue_delay, 0.95),
                "p99_queue_delay_s": empirical_quantile(queue_delay, 0.99),
                "queue_probability": float((queue_delay > 0).mean()),
                "utilization": float(work.sum() / (duration_s * service_rate)),
            }
        ]
    )

    slack_table = bucket_required_units(
        frame,
        profile,
        units=units,
        windows_s=windows_s,
        output_token_source=options.output_token_source,
    )
    slack_summary = summarize_slack(slack_table)

    assumptions = {
        "output_token_source": options.output_token_source,
        "baseline_latency_model": baseline_model,
        "service_rate_adjusted_tokens_per_s": service_rate,
        "windows_s": tuple(float(window) for window in windows_s),
    }
    return SimulationResult(
        units=units,
        unit_name=profile.unit_name,
        request_level=frame,
        latency_summary=latency_summary,
        slack_summary=slack_summary,
        assumptions=assumptions,
    )
