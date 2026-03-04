"""Core data structures for capacity planning.

This module defines the schema classes used throughout slosizer for representing
request traces, capacity profiles, SLO targets, and planning results.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

Percentile = float


@dataclass(frozen=True)
class RequestSchema:
    """Column mapping for request trace DataFrames.

    Attributes:
        time_col: Column containing request arrival timestamps.
        class_col: Column containing request class labels.
        input_tokens_col: Column containing input token counts.
        cached_input_tokens_col: Column containing cached input token counts.
        output_tokens_col: Column containing output token counts.
        thinking_tokens_col: Column containing thinking/reasoning token counts.
        max_output_tokens_col: Column containing max output token limits.
        latency_col: Column containing observed latency in seconds.
    """

    time_col: str = "ts"
    class_col: str | None = "class_name"
    input_tokens_col: str = "input_tokens"
    cached_input_tokens_col: str | None = "cached_input_tokens"
    output_tokens_col: str = "output_tokens"
    thinking_tokens_col: str | None = "thinking_tokens"
    max_output_tokens_col: str | None = "max_output_tokens"
    latency_col: str | None = "latency_s"


@dataclass(frozen=True)
class RequestTrace:
    """Normalized request trace with canonical columns.

    Attributes:
        frame: DataFrame with canonical columns (arrival_s, input_tokens, etc.).
        schema: Original schema used to parse the trace.
        provider: Cloud provider name (e.g., "vertex", "azure").
        model: Model identifier.
        region: Deployment region.
        metadata: Additional trace metadata.
    """

    frame: pd.DataFrame
    schema: RequestSchema
    provider: str | None = None
    model: str | None = None
    region: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapacityProfile:
    """Provider-specific capacity configuration.

    Defines how tokens translate to reserved capacity units and the constraints
    on purchasing those units.

    Attributes:
        provider: Cloud provider name.
        model: Model identifier.
        unit_name: Name of capacity unit (e.g., "GSU", "PTU").
        throughput_per_unit: Tokens per second per capacity unit.
        purchase_increment: Minimum increment for purchasing units.
        min_units: Minimum number of units that can be provisioned.
        input_weight: Token weight multiplier for input tokens.
        cached_input_weight: Token weight multiplier for cached input tokens.
        output_weight: Token weight multiplier for output tokens.
        thinking_weight: Token weight multiplier for thinking tokens.
        long_input_threshold: Input token count above which long-context weights apply.
        long_input_input_weight: Input weight for long-context requests.
        long_input_cached_input_weight: Cached input weight for long-context requests.
        long_input_output_weight: Output weight for long-context requests.
        long_input_thinking_weight: Thinking weight for long-context requests.
        source: Documentation or calibration source for the profile.
        notes: Additional notes about the profile.
    """

    provider: str
    model: str
    unit_name: Literal["GSU", "PTU", "capacity_unit"]
    throughput_per_unit: float | None
    purchase_increment: int = 1
    min_units: int = 1
    input_weight: float = 1.0
    cached_input_weight: float = 0.0
    output_weight: float = 4.0
    thinking_weight: float = 4.0
    long_input_threshold: int | None = None
    long_input_input_weight: float | None = None
    long_input_cached_input_weight: float | None = None
    long_input_output_weight: float | None = None
    long_input_thinking_weight: float | None = None
    source: str = ""
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LatencySLO:
    """Latency service level objective.

    Attributes:
        threshold_s: Maximum acceptable latency in seconds.
        percentile: Target percentile (e.g., 0.99 for p99).
        metric: Latency metric to measure ("e2e" or "queue_delay").
    """

    threshold_s: float
    percentile: Percentile = 0.99
    metric: Literal["e2e", "queue_delay"] = "e2e"


@dataclass(frozen=True)
class ThroughputTarget:
    """Throughput-based capacity planning target.

    Attributes:
        percentile: Target percentile for required capacity.
        max_overload_probability: Maximum acceptable probability of overload.
        windows_s: Time window sizes for bucket analysis.
    """

    percentile: Percentile | None = 0.99
    max_overload_probability: float | None = None
    windows_s: tuple[float, ...] = (1.0, 5.0, 30.0)

    def label(self) -> str:
        """Generate a human-readable label for this target.

        Returns:
            Descriptive label string.
        """
        parts = ["throughput"]
        if self.percentile is not None:
            parts.append(f"p{int(self.percentile * 100)}")
        if self.max_overload_probability is not None:
            parts.append(f"overload<={self.max_overload_probability:.3f}")
        return "-".join(parts)


@dataclass(frozen=True)
class LatencyTarget:
    """Latency-based capacity planning target.

    Attributes:
        slo: The latency SLO to meet.
    """

    slo: LatencySLO

    def label(self) -> str:
        """Generate a human-readable label for this target.

        Returns:
            Descriptive label string.
        """
        percentile = int(self.slo.percentile * 100)
        return f"latency-p{percentile}<={self.slo.threshold_s:.3f}s"


@dataclass(frozen=True)
class BaselineLatencyModel:
    """Linear model for baseline request latency.

    Predicts latency as a linear combination of token counts, useful for
    estimating processing time independent of queueing.

    Attributes:
        intercept_s: Base latency in seconds.
        input_token_s: Seconds per input token.
        cached_input_token_s: Seconds per cached input token.
        output_token_s: Seconds per output token.
        thinking_token_s: Seconds per thinking token.
    """

    intercept_s: float = 0.15
    input_token_s: float = 3.0e-5
    cached_input_token_s: float = 8.0e-6
    output_token_s: float = 9.0e-4
    thinking_token_s: float = 7.0e-4

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict baseline latency for each request.

        Args:
            frame: DataFrame with token count columns.

        Returns:
            Array of predicted latencies in seconds.
        """
        values = (
            self.intercept_s
            + self.input_token_s * frame["input_tokens"].to_numpy(dtype=float)
            + self.cached_input_token_s * frame["cached_input_tokens"].to_numpy(dtype=float)
            + self.output_token_s * frame["output_tokens"].to_numpy(dtype=float)
            + self.thinking_token_s * frame["thinking_tokens"].to_numpy(dtype=float)
        )
        return np.clip(values, 1e-6, None)


@dataclass(frozen=True)
class PlanOptions:
    """Options for capacity planning.

    Attributes:
        output_token_source: Use "observed" or "max_output_tokens" for planning.
        max_units_to_search: Maximum capacity units to consider during search.
        headroom_factor: Additional capacity buffer as a fraction (e.g., 0.1 for 10%).
        baseline_latency_model: Custom latency model; if None, one is fitted.
    """

    output_token_source: Literal["observed", "max_output_tokens"] = "observed"
    max_units_to_search: int = 200
    headroom_factor: float = 0.0
    baseline_latency_model: BaselineLatencyModel | None = None


@dataclass
class SimulationResult:
    """Results from a capacity simulation.

    Attributes:
        units: Number of capacity units simulated.
        unit_name: Name of capacity unit.
        request_level: Per-request simulation results.
        latency_summary: Aggregate latency statistics.
        slack_summary: Spare capacity statistics by time window.
        assumptions: Simulation parameters and settings.
    """

    units: int
    unit_name: str
    request_level: pd.DataFrame
    latency_summary: pd.DataFrame
    slack_summary: pd.DataFrame
    assumptions: dict[str, Any]


@dataclass
class PlanResult:
    """Results from capacity planning.

    Attributes:
        objective: Planning objective ("throughput" or "latency").
        target: Human-readable target description.
        recommended_units: Recommended number of capacity units.
        unit_name: Name of capacity unit.
        metrics: Planning metrics and statistics.
        slack_summary: Spare capacity statistics.
        latency_summary: Latency statistics (for latency planning).
        request_level: Per-request results (for latency planning).
        assumptions: Planning parameters and settings.
    """

    objective: str
    target: str
    recommended_units: int
    unit_name: str
    metrics: dict[str, Any]
    slack_summary: pd.DataFrame
    latency_summary: pd.DataFrame | None = None
    request_level: pd.DataFrame | None = None
    assumptions: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Convert result to a flat dictionary.

        Returns:
            Dictionary with all metrics and metadata.
        """
        row = dict(self.metrics)
        row.update(
            {
                "objective": self.objective,
                "target": self.target,
                "recommended_units": self.recommended_units,
                "unit_name": self.unit_name,
            }
        )
        return row
