"""Visualization functions for capacity planning results."""

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from slosizer.schema import CapacityProfile, LatencyTarget, PlanOptions, RequestTrace
from slosizer.simulation import bucket_required_units, simulate_capacity


def _maybe_save(path: str | Path | None) -> None:
    """Save figure to path if provided."""
    if path is not None:
        plt.savefig(path, dpi=180, bbox_inches="tight")


def plot_latency_vs_units(
    trace: RequestTrace,
    profile: CapacityProfile,
    *,
    units: Iterable[int],
    options: PlanOptions | None = None,
    target: LatencyTarget | None = None,
    path: str | Path | None = None,
) -> None:
    """Plot latency percentiles as a function of reserved capacity.

    Args:
        trace: Request trace to simulate.
        profile: Capacity profile.
        units: Capacity unit values to plot.
        options: Planning options.
        target: Optional latency target to show as horizontal line.
        path: Optional path to save the figure.
    """
    if options is None:
        options = PlanOptions()

    rows = []
    for unit in units:
        simulation = simulate_capacity(trace, profile, units=int(unit), options=options)
        summary = simulation.latency_summary.iloc[0]
        rows.append(
            {
                "units": int(unit),
                "p95_latency_s": float(summary["p95_latency_s"]),
                "p99_latency_s": float(summary["p99_latency_s"]),
            }
        )
    plot_df = pd.DataFrame(rows)
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(plot_df["units"], plot_df["p95_latency_s"], marker="o", label="p95 latency")
    plt.plot(plot_df["units"], plot_df["p99_latency_s"], marker="s", label="p99 latency")
    if target is not None:
        plt.axhline(target.slo.threshold_s, linestyle="--", label="target latency")
    plt.xlabel(f"Provisioned {profile.unit_name}s")
    plt.ylabel("Latency (seconds)")
    plt.title("Latency vs provisioned capacity")
    plt.legend()
    _maybe_save(path)
    plt.close()


def plot_required_units_distribution(
    trace: RequestTrace,
    profile: CapacityProfile,
    *,
    windows_s: tuple[float, ...] = (1.0, 5.0, 30.0),
    options: PlanOptions | None = None,
    path: str | Path | None = None,
) -> None:
    """Plot histogram of required capacity units per time window.

    Args:
        trace: Request trace to analyze.
        profile: Capacity profile.
        windows_s: Time window sizes to plot.
        options: Planning options.
        path: Optional path to save the figure.
    """
    if options is None:
        options = PlanOptions()

    required = bucket_required_units(
        trace.frame,
        profile,
        units=0,
        windows_s=windows_s,
        output_token_source=options.output_token_source,
    )
    plt.figure(figsize=(8.5, 5.2))
    for window_s in windows_s:
        subset = required[required["window_s"] == float(window_s)]
        plt.hist(subset["required_units"], bins=30, alpha=0.6, label=f"{window_s:g}s window")
    plt.xlabel(f"Required {profile.unit_name}s")
    plt.ylabel("Bucket count")
    plt.title("Distribution of required reserved capacity")
    plt.legend()
    _maybe_save(path)
    plt.close()


def plot_capacity_tradeoff(comparison: pd.DataFrame, *, path: str | Path | None = None) -> None:
    """Plot recommended capacity across scenarios and targets.

    Args:
        comparison: Output from compare_scenarios.
        path: Optional path to save the figure.
    """
    labels = comparison["scenario"] + "\n" + comparison["target"]
    plt.figure(figsize=(10, 5.5))
    plt.bar(labels, comparison["recommended_units"])
    plt.ylabel("Recommended units")
    plt.xlabel("Scenario / target")
    plt.title("Recommended capacity by scenario and planning objective")
    plt.xticks(rotation=20, ha="right")
    _maybe_save(path)
    plt.close()


def plot_slack_tradeoff(comparison: pd.DataFrame, *, path: str | Path | None = None) -> None:
    """Plot spare capacity fraction across scenarios and targets.

    Args:
        comparison: Output from compare_scenarios.
        path: Optional path to save the figure.

    Raises:
        ValueError: If comparison is missing the avg_spare_fraction_1s column.
    """
    if "avg_spare_fraction_1s" not in comparison.columns:
        raise ValueError("comparison DataFrame must contain 'avg_spare_fraction_1s' column")
    labels = comparison["scenario"] + "\n" + comparison["target"]
    plt.figure(figsize=(10, 5.5))
    plt.bar(labels, comparison["avg_spare_fraction_1s"])
    plt.ylabel("Average spare fraction (1s window)")
    plt.xlabel("Scenario / target")
    plt.title("Slack capacity trade-off")
    plt.xticks(rotation=20, ha="right")
    _maybe_save(path)
    plt.close()
