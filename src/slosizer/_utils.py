"""Internal utility functions for capacity calculations."""

import math

import numpy as np
import pandas as pd

from slosizer.schema import CapacityProfile


def round_up_to_increment(units: float, minimum: int, increment: int) -> int:
    """Round capacity units up to the nearest valid purchase amount.

    Args:
        units: Raw number of units needed.
        minimum: Minimum allowed units.
        increment: Purchase increment constraint.

    Returns:
        Rounded-up unit count satisfying constraints.
    """
    base = max(float(minimum), units)
    return int(math.ceil(base / increment) * increment)


def selected_output_tokens(frame: pd.DataFrame, source: str) -> np.ndarray:
    """Select output token values based on source preference.

    Args:
        frame: DataFrame with output_tokens and max_output_tokens columns.
        source: Either "observed" or "max_output_tokens".

    Returns:
        Array of output token counts.
    """
    if source == "max_output_tokens":
        return frame["max_output_tokens"].to_numpy(dtype=float)
    return frame["output_tokens"].to_numpy(dtype=float)


def adjusted_work(
    frame: pd.DataFrame,
    profile: CapacityProfile,
    *,
    output_token_source: str = "observed",
) -> np.ndarray:
    """Compute weighted token work for each request.

    Applies provider-specific token weights and long-context adjustments
    to compute the effective work each request contributes to capacity.

    Args:
        frame: DataFrame with token count columns.
        profile: Capacity profile with token weights.
        output_token_source: Source for output tokens ("observed" or "max_output_tokens").

    Returns:
        Array of adjusted work values per request.
    """
    inputs = frame["input_tokens"].to_numpy(dtype=float)
    cached = frame["cached_input_tokens"].to_numpy(dtype=float)
    outputs = selected_output_tokens(frame, output_token_source)
    thinking = frame["thinking_tokens"].to_numpy(dtype=float)

    input_weight = np.full(len(frame), profile.input_weight, dtype=float)
    cached_weight = np.full(len(frame), profile.cached_input_weight, dtype=float)
    output_weight = np.full(len(frame), profile.output_weight, dtype=float)
    thinking_weight = np.full(len(frame), profile.thinking_weight, dtype=float)

    if profile.long_input_threshold is not None:
        mask = inputs > float(profile.long_input_threshold)
        if profile.long_input_input_weight is not None:
            input_weight = np.where(mask, profile.long_input_input_weight, input_weight)
        if profile.long_input_cached_input_weight is not None:
            cached_weight = np.where(mask, profile.long_input_cached_input_weight, cached_weight)
        if profile.long_input_output_weight is not None:
            output_weight = np.where(mask, profile.long_input_output_weight, output_weight)
        if profile.long_input_thinking_weight is not None:
            thinking_weight = np.where(mask, profile.long_input_thinking_weight, thinking_weight)

    return (
        (inputs - cached) * input_weight
        + cached * cached_weight
        + outputs * output_weight
        + thinking * thinking_weight
    )
