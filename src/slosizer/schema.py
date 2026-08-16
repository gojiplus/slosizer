"""Core data structures for capacity planning.

This module defines the schema classes used throughout slosizer for representing
request traces, capacity profiles, SLO targets, and planning results.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any, Literal

import numpy as np
import pandas as pd

Percentile = float
HybridStrategy = Literal["cost_optimal", "percentile_split"]
SLOPolicy = Literal["hard", "priced"]


class OutputTokenSource(StrEnum):
    """Source for output token counts in capacity planning.

    Attributes:
        OBSERVED: Use actual observed output token counts from trace data.
        MAX_OUTPUT_TOKENS: Use max_output_tokens limit for worst-case planning.
    """

    OBSERVED = "observed"
    MAX_OUTPUT_TOKENS = "max_output_tokens"


class LatencyMetric(StrEnum):
    """Latency metric for SLO evaluation.

    Attributes:
        E2E: End-to-end latency including baseline model latency and queue delay.
        QUEUE_DELAY: Queue delay only, excluding baseline model latency.
    """

    E2E = "e2e"
    QUEUE_DELAY = "queue_delay"


@dataclass(frozen=True)
class RequestSchema:
    """Column mapping for request trace DataFrames.

    Attributes:
        time_col: Column containing request arrival timestamps.
        class_col: Column containing request class labels.
        input_tokens_col: Column containing total input tokens, including cached input.
        cached_input_tokens_col: Column containing the cached subset of input tokens.
        output_tokens_col: Column containing non-reasoning response tokens.
        thinking_tokens_col: Column containing additional thinking/reasoning tokens
            not included in ``output_tokens_col``.
        max_output_tokens_col: Column containing max output token limits.
        latency_col: Column containing observed latency in seconds.
        request_id_col: Column containing the application request identifier.
        request_model_col: Column containing the model requested by the client.
        response_model_col: Column containing the model that served the request.
        service_tier_col: Column containing the provider service tier or deployment.
        business_value_col: Column containing expected gross business value per request.
    """

    time_col: str = "ts"
    class_col: str | None = "class_name"
    input_tokens_col: str = "input_tokens"
    cached_input_tokens_col: str | None = "cached_input_tokens"
    output_tokens_col: str = "output_tokens"
    thinking_tokens_col: str | None = "thinking_tokens"
    max_output_tokens_col: str | None = "max_output_tokens"
    latency_col: str | None = "latency_s"
    request_id_col: str | None = "request_id"
    request_model_col: str | None = "request_model"
    response_model_col: str | None = "response_model"
    service_tier_col: str | None = "service_tier"
    business_value_col: str | None = "business_value"


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
        deployment_type: Provider deployment or capacity offering.
        region: Region to which the profile applies, if region-specific.
        effective_from: Date on which the provider facts became effective.
        verified_on: Date on which the source was last checked.
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
    deployment_type: str = "provisioned"
    region: str | None = None
    effective_from: date | None = None
    verified_on: date | None = None

    def __post_init__(self) -> None:
        if self.throughput_per_unit is not None and self.throughput_per_unit <= 0:
            raise ValueError(
                f"throughput_per_unit must be positive, got {self.throughput_per_unit}"
            )
        if self.purchase_increment < 1:
            raise ValueError(
                f"purchase_increment must be >= 1, got {self.purchase_increment}"
            )
        if self.min_units < 1:
            raise ValueError(f"min_units must be >= 1, got {self.min_units}")
        if self.input_weight < 0:
            raise ValueError(
                f"input_weight must be non-negative, got {self.input_weight}"
            )
        if self.cached_input_weight < 0:
            raise ValueError(
                f"cached_input_weight must be non-negative, got {self.cached_input_weight}"
            )
        if self.output_weight < 0:
            raise ValueError(
                f"output_weight must be non-negative, got {self.output_weight}"
            )
        if self.thinking_weight < 0:
            raise ValueError(
                f"thinking_weight must be non-negative, got {self.thinking_weight}"
            )
        if self.long_input_threshold is not None and self.long_input_threshold <= 0:
            raise ValueError("long_input_threshold must be positive")
        long_weights = (
            self.long_input_input_weight,
            self.long_input_cached_input_weight,
            self.long_input_output_weight,
            self.long_input_thinking_weight,
        )
        if any(weight is not None and weight < 0 for weight in long_weights):
            raise ValueError("long-context token weights must be non-negative")


@dataclass(frozen=True)
class LatencySLO:
    """Latency service level objective.

    Attributes:
        threshold_s: Maximum acceptable latency in seconds.
        percentile: Target percentile (e.g., 0.99 for p99).
        metric: Latency metric to measure (E2E or QUEUE_DELAY).

    Raises:
        ValueError: If threshold_s <= 0 or percentile not in (0, 1).
    """

    threshold_s: float
    percentile: Percentile = 0.99
    metric: LatencyMetric = LatencyMetric.E2E

    def __post_init__(self) -> None:
        if self.threshold_s <= 0:
            raise ValueError(f"threshold_s must be positive, got {self.threshold_s}")
        if not (0 < self.percentile < 1):
            raise ValueError(f"percentile must be in (0, 1), got {self.percentile}")


@dataclass(frozen=True)
class ThroughputTarget:
    """Throughput-based capacity planning target.

    Attributes:
        percentile: Target percentile for required capacity.
        max_overload_probability: Maximum acceptable probability of overload.
        windows_s: Time window sizes for bucket analysis.

    Raises:
        ValueError: If percentile not in (0, 1) or max_overload_probability not in [0, 1].
    """

    percentile: Percentile | None = 0.99
    max_overload_probability: float | None = None
    windows_s: tuple[float, ...] = (1.0, 5.0, 30.0)

    def __post_init__(self) -> None:
        if self.percentile is not None and not (0 < self.percentile < 1):
            raise ValueError(f"percentile must be in (0, 1), got {self.percentile}")
        if self.max_overload_probability is not None and not (
            0 <= self.max_overload_probability <= 1
        ):
            raise ValueError(
                f"max_overload_probability must be in [0, 1], got {self.max_overload_probability}"
            )
        if not self.windows_s:
            raise ValueError("windows_s must not be empty")
        if any(w <= 0 for w in self.windows_s):
            raise ValueError("all windows_s values must be positive")

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
            + self.cached_input_token_s
            * frame["cached_input_tokens"].to_numpy(dtype=float)
            + self.output_token_s * frame["output_tokens"].to_numpy(dtype=float)
            + self.thinking_token_s * frame["thinking_tokens"].to_numpy(dtype=float)
        )
        return np.clip(values, 1e-6, None)


@dataclass(frozen=True)
class PlanOptions:
    """Options for capacity planning.

    Attributes:
        output_token_source: Use OBSERVED or MAX_OUTPUT_TOKENS for planning.
        max_units_to_search: Maximum capacity units to consider during search.
        headroom_factor: Additional capacity buffer as a fraction (e.g., 0.1 for 10%).
        baseline_latency_model: Custom latency model; if None, one is fitted.

    Raises:
        ValueError: If max_units_to_search < 1 or headroom_factor < 0.
    """

    output_token_source: OutputTokenSource = OutputTokenSource.OBSERVED
    max_units_to_search: int = 200
    headroom_factor: float = 0.0
    baseline_latency_model: BaselineLatencyModel | None = None

    def __post_init__(self) -> None:
        if self.max_units_to_search < 1:
            raise ValueError(
                f"max_units_to_search must be >= 1, got {self.max_units_to_search}"
            )
        if self.headroom_factor < 0:
            raise ValueError(
                f"headroom_factor must be non-negative, got {self.headroom_factor}"
            )


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


@dataclass(frozen=True)
class PaygoPricing:
    """Per-token pricing for overflow traffic.

    Attributes:
        input_cost_per_million: Cost per million input tokens.
        output_cost_per_million: Cost per million output tokens.
        cached_input_cost_per_million: Cost per million cached input tokens. If
            omitted, the regular input price is used.
        thinking_cost_per_million: Cost per million separately reported reasoning
            tokens. If omitted, the regular output price is used.

    Raises:
        ValueError: If costs are negative.
    """

    input_cost_per_million: float
    output_cost_per_million: float
    cached_input_cost_per_million: float | None = None
    thinking_cost_per_million: float | None = None

    def __post_init__(self) -> None:
        prices = {
            "input_cost_per_million": self.input_cost_per_million,
            "output_cost_per_million": self.output_cost_per_million,
            "cached_input_cost_per_million": self.cached_input_cost_per_million,
            "thinking_cost_per_million": self.thinking_cost_per_million,
        }
        for name, value in prices.items():
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.input_cost_per_million < 0:
            raise ValueError(
                f"input_cost_per_million must be non-negative, got {self.input_cost_per_million}"
            )
        if self.output_cost_per_million < 0:
            raise ValueError(
                f"output_cost_per_million must be non-negative, got {self.output_cost_per_million}"
            )
        if (
            self.cached_input_cost_per_million is not None
            and self.cached_input_cost_per_million < 0
        ):
            raise ValueError(
                "cached_input_cost_per_million must be non-negative, "
                f"got {self.cached_input_cost_per_million}"
            )
        if (
            self.thinking_cost_per_million is not None
            and self.thinking_cost_per_million < 0
        ):
            raise ValueError(
                "thinking_cost_per_million must be non-negative, "
                f"got {self.thinking_cost_per_million}"
            )

    @property
    def effective_cached_input_cost_per_million(self) -> float:
        """Return the cached-input price with a conservative fallback."""
        if self.cached_input_cost_per_million is None:
            return self.input_cost_per_million
        return self.cached_input_cost_per_million

    @property
    def effective_thinking_cost_per_million(self) -> float:
        """Return the reasoning-token price with a conservative fallback."""
        if self.thinking_cost_per_million is None:
            return self.output_cost_per_million
        return self.thinking_cost_per_million


@dataclass(frozen=True)
class ProvisionedPricing:
    """Hourly cost for provisioned capacity.

    Attributes:
        cost_per_unit_hour: Cost per capacity unit per hour.

    Raises:
        ValueError: If cost is negative.
    """

    cost_per_unit_hour: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.cost_per_unit_hour):
            raise ValueError("cost_per_unit_hour must be finite")
        if self.cost_per_unit_hour < 0:
            raise ValueError(
                f"cost_per_unit_hour must be non-negative, got {self.cost_per_unit_hour}"
            )


@dataclass(frozen=True)
class RateCard:
    """Effective-dated model and deployment prices.

    Attributes:
        provisioned: Hourly cost for provisioned capacity.
        paygo: Per-token pricing for overflow traffic, if applicable.
        currency: ISO 4217 currency code for all prices.
        provider: Provider to which the rate card applies.
        model: Model to which the rate card applies.
        region: Region to which the rate card applies.
        deployment_type: Provider deployment or capacity offering.
        effective_from: First date on which the rate card applies.
        effective_to: Last date on which the rate card applies.
        verified_on: Date on which the price source was last checked.
        source: Contract, invoice, or public rate-card source.
    """

    provisioned: ProvisionedPricing
    paygo: PaygoPricing | None = None
    currency: str = "USD"
    provider: str | None = None
    model: str | None = None
    region: str | None = None
    deployment_type: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    verified_on: date | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if len(self.currency) != 3 or not self.currency.isalpha():
            raise ValueError("currency must be a three-letter ISO 4217 code")
        object.__setattr__(self, "currency", self.currency.upper())
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_to < self.effective_from
        ):
            raise ValueError("effective_to must not precede effective_from")

    def validate_for(self, profile: CapacityProfile) -> None:
        """Validate that explicitly scoped prices match a capacity profile."""
        checks = (
            ("provider", self.provider, profile.provider),
            ("model", self.model, profile.model),
            ("region", self.region, profile.region),
            ("deployment_type", self.deployment_type, profile.deployment_type),
        )
        for field_name, priced_value, profile_value in checks:
            if priced_value is not None and priced_value != profile_value:
                raise ValueError(
                    f"pricing {field_name} {priced_value!r} does not match "
                    f"profile {field_name} {profile_value!r}"
                )


@dataclass(frozen=True)
class HybridTarget:
    """Target for hybrid capacity planning.

    Attributes:
        strategy: Planning strategy - "cost_optimal" or "percentile_split".
        provision_percentile: Percentile to provision for (required if strategy="percentile_split").
        latency_slo: Optional latency SLO constraint.

    Raises:
        ValueError: If strategy is "percentile_split" but provision_percentile is not set,
            or if provision_percentile is not in (0, 1).
    """

    strategy: HybridStrategy
    provision_percentile: float | None = None
    latency_slo: "LatencySLO | None" = None

    def __post_init__(self) -> None:
        if self.strategy not in ("cost_optimal", "percentile_split"):
            raise ValueError(f"unknown hybrid strategy {self.strategy!r}")
        if self.strategy == "percentile_split" and self.provision_percentile is None:
            raise ValueError(
                "provision_percentile is required when strategy='percentile_split'"
            )
        if self.provision_percentile is not None and not (
            0 < self.provision_percentile < 1
        ):
            raise ValueError(
                f"provision_percentile must be in (0, 1), got {self.provision_percentile}"
            )

    def label(self) -> str:
        """Generate a human-readable label for this target.

        Returns:
            Descriptive label string.
        """
        if self.strategy == "cost_optimal":
            label = "hybrid-cost-optimal"
        else:
            pct = (
                int(self.provision_percentile * 100) if self.provision_percentile else 0
            )
            label = f"hybrid-p{pct}-split"
        if self.latency_slo is not None:
            label += f"-slo-p{int(self.latency_slo.percentile * 100)}<={self.latency_slo.threshold_s:.1f}s"
        return label


@dataclass
class HybridPlanResult:
    """Results from hybrid capacity planning.

    Attributes:
        provisioned_units: Number of provisioned capacity units.
        unit_name: Name of capacity unit (e.g., "GSU", "PTU").
        currency: ISO 4217 currency code for financial values.
        provisioned_cost_hourly: Hourly cost of provisioned capacity.
        paygo_cost_hourly: Hourly cost of overflow to paygo.
        total_cost_hourly: Total hourly cost (provisioned + paygo).
        full_provision_units: Units needed if provisioning for 100% of traffic.
        full_provision_cost_hourly: Hourly cost if fully provisioned.
        savings_vs_full_provision: Dollar savings per hour vs full provision.
        savings_percent: Percentage savings vs full provision.
        overflow_fraction: Fraction of time buckets with overflow.
        overflow_input_tokens_hourly: Average overflow input tokens per hour.
        overflow_cached_input_tokens_hourly: Average cached input overflow per hour.
        overflow_output_tokens_hourly: Average overflow output tokens per hour.
        overflow_thinking_tokens_hourly: Average reasoning-token overflow per hour.
        slack_summary: Spare capacity statistics by time window.
        assumptions: Planning parameters and settings.
    """

    provisioned_units: int
    unit_name: str
    currency: str
    provisioned_cost_hourly: float
    paygo_cost_hourly: float
    total_cost_hourly: float
    full_provision_units: int
    full_provision_cost_hourly: float
    savings_vs_full_provision: float
    savings_percent: float
    overflow_fraction: float
    overflow_input_tokens_hourly: float
    overflow_cached_input_tokens_hourly: float
    overflow_output_tokens_hourly: float
    overflow_thinking_tokens_hourly: float
    slack_summary: pd.DataFrame
    assumptions: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Convert result to a flat dictionary.

        Returns:
            Dictionary with all metrics and metadata.
        """
        return {
            "provisioned_units": self.provisioned_units,
            "unit_name": self.unit_name,
            "currency": self.currency,
            "provisioned_cost_hourly": self.provisioned_cost_hourly,
            "paygo_cost_hourly": self.paygo_cost_hourly,
            "total_cost_hourly": self.total_cost_hourly,
            "full_provision_units": self.full_provision_units,
            "full_provision_cost_hourly": self.full_provision_cost_hourly,
            "savings_vs_full_provision": self.savings_vs_full_provision,
            "savings_percent": self.savings_percent,
            "overflow_fraction": self.overflow_fraction,
            "overflow_input_tokens_hourly": self.overflow_input_tokens_hourly,
            "overflow_cached_input_tokens_hourly": self.overflow_cached_input_tokens_hourly,
            "overflow_output_tokens_hourly": self.overflow_output_tokens_hourly,
            "overflow_thinking_tokens_hourly": self.overflow_thinking_tokens_hourly,
        }


@dataclass(frozen=True)
class ProfitTarget:
    """Optional absolute-profit or priced-SLO objective.

    ``business_value`` is gross contribution before inference and SLO costs. It
    can be supplied as one value for every request or through the canonical
    ``business_value`` trace column. It does not change the request count or
    timing. Use ``HybridTarget`` for fixed-demand cost optimization under a hard
    SLO; that common case does not require business value. With one trace and a
    hard SLO, business value changes reported profit but not recommended
    capacity.

    Attributes:
        latency_slo: Latency promise used to classify good and bad requests.
        slo_policy: ``hard`` excludes plans that miss the SLO; ``priced`` assigns
            the stated penalty to every bad request.
        value_per_request: Expected gross value for each request. If omitted, the
            trace must contain a complete ``business_value`` column.
        slo_violation_cost_per_request: Business cost of a request that exceeds
            the latency threshold. Required when ``slo_policy`` is ``priced``.
    """

    latency_slo: LatencySLO
    slo_policy: SLOPolicy = "hard"
    value_per_request: float | None = None
    slo_violation_cost_per_request: float = 0.0

    def __post_init__(self) -> None:
        if self.slo_policy not in ("hard", "priced"):
            raise ValueError(f"unknown SLO policy {self.slo_policy!r}")
        if self.value_per_request is not None and not np.isfinite(
            self.value_per_request
        ):
            raise ValueError("value_per_request must be finite")
        if self.value_per_request is not None and self.value_per_request < 0:
            raise ValueError("value_per_request must be non-negative")
        if not np.isfinite(self.slo_violation_cost_per_request):
            raise ValueError("slo_violation_cost_per_request must be finite")
        if self.slo_violation_cost_per_request < 0:
            raise ValueError("slo_violation_cost_per_request must be non-negative")
        if self.slo_policy == "priced" and self.slo_violation_cost_per_request == 0:
            raise ValueError(
                "slo_violation_cost_per_request must be positive when "
                "slo_policy='priced'"
            )


@dataclass
class ProfitPlanResult:
    """Profit-maximizing reserved-capacity plan and its candidate frontier."""

    recommended_units: int
    unit_name: str
    currency: str
    gross_value_hourly: float
    provisioned_cost_hourly: float
    slo_violation_cost_hourly: float
    expected_profit_hourly: float
    slo_attainment: float
    candidate_plans: pd.DataFrame
    assumptions: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Convert the recommended plan to a flat dictionary."""
        return {
            "recommended_units": self.recommended_units,
            "unit_name": self.unit_name,
            "currency": self.currency,
            "gross_value_hourly": self.gross_value_hourly,
            "provisioned_cost_hourly": self.provisioned_cost_hourly,
            "slo_violation_cost_hourly": self.slo_violation_cost_hourly,
            "expected_profit_hourly": self.expected_profit_hourly,
            "slo_attainment": self.slo_attainment,
        }


@dataclass(frozen=True)
class ProfitScenario:
    """Named model forecast for optional cross-model economic comparison.

    Each trace is an external demand forecast. The package does not estimate
    how a model changes request count, timing, token mix, or burstiness.
    """

    name: str
    trace: RequestTrace
    profile: CapacityProfile
    pricing: RateCard
    target: ProfitTarget
