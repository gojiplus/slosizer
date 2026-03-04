"""Request trace ingestion and normalization.

This module provides functions to convert raw DataFrames into normalized
RequestTrace objects with canonical column names.
"""

from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from slosizer.schema import RequestSchema, RequestTrace


def _normalize_arrival_seconds(series: pd.Series) -> pd.Series:
    """Convert arrival times to seconds from trace start.

    Args:
        series: Arrival time column (datetime, numeric, or string).

    Returns:
        Series of float seconds relative to the first arrival.
    """
    if is_datetime64_any_dtype(series):
        return (series - series.min()).dt.total_seconds().astype(float)
    if is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="raise").astype(float)
        return values - float(values.min())

    parsed = pd.to_datetime(series, errors="raise")
    return (parsed - parsed.min()).dt.total_seconds().astype(float)


def _coerce_nonnegative_numeric(
    data: pd.DataFrame,
    source_col: str | None,
    *,
    default: float = 0.0,
) -> pd.Series:
    """Extract and validate a non-negative numeric column.

    Args:
        data: Source DataFrame.
        source_col: Column name to extract.
        default: Default value if column is missing.

    Returns:
        Series of non-negative float values.

    Raises:
        ValueError: If column contains missing, non-numeric, or negative values.
    """
    if source_col is None or source_col not in data.columns:
        return pd.Series(default, index=data.index, dtype=float)
    values = pd.to_numeric(data[source_col], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Column {source_col!r} contains missing or non-numeric values.")
    if (values < 0).any():
        raise ValueError(f"Column {source_col!r} contains negative values.")
    return values.astype(float)


def from_dataframe(
    df: pd.DataFrame,
    *,
    schema: RequestSchema,
    provider: str | None = None,
    model: str | None = None,
    region: str | None = None,
    validate: bool = True,
    metadata: dict[str, Any] | None = None,
) -> RequestTrace:
    """Create a RequestTrace from a DataFrame.

    Normalizes column names and validates data according to the schema.

    Args:
        df: Source DataFrame with request data.
        schema: Column mapping for the DataFrame.
        provider: Cloud provider name.
        model: Model identifier.
        region: Deployment region.
        validate: Whether to validate data constraints.
        metadata: Additional trace metadata.

    Returns:
        Normalized RequestTrace.

    Raises:
        ValueError: If required columns are missing or validation fails.
    """
    required_columns = [
        schema.time_col,
        schema.input_tokens_col,
        schema.output_tokens_col,
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_str}")

    raw = df.copy()
    frame = raw.copy()

    frame["arrival_s"] = _normalize_arrival_seconds(raw[schema.time_col])
    frame["class_name"] = (
        raw[schema.class_col].astype(str)
        if schema.class_col and schema.class_col in raw.columns
        else "default"
    )
    frame["input_tokens"] = (
        _coerce_nonnegative_numeric(raw, schema.input_tokens_col).round().astype(int)
    )
    frame["cached_input_tokens"] = (
        _coerce_nonnegative_numeric(
            raw,
            schema.cached_input_tokens_col,
        )
        .round()
        .astype(int)
    )
    frame["output_tokens"] = (
        _coerce_nonnegative_numeric(raw, schema.output_tokens_col).round().astype(int)
    )
    frame["thinking_tokens"] = (
        _coerce_nonnegative_numeric(
            raw,
            schema.thinking_tokens_col,
        )
        .round()
        .astype(int)
    )
    default_max_output = float(frame["output_tokens"].max()) if len(frame) else 0.0
    frame["max_output_tokens"] = (
        _coerce_nonnegative_numeric(
            raw,
            schema.max_output_tokens_col,
            default=default_max_output,
        )
        .round()
        .astype(int)
    )

    if schema.latency_col and schema.latency_col in raw.columns:
        frame["observed_latency_s"] = _coerce_nonnegative_numeric(raw, schema.latency_col)
    else:
        frame["observed_latency_s"] = pd.Series(float("nan"), index=frame.index)

    canonical_columns = [
        "arrival_s",
        "class_name",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "thinking_tokens",
        "max_output_tokens",
        "observed_latency_s",
    ]
    frame = (
        frame[canonical_columns].sort_values("arrival_s", kind="mergesort").reset_index(drop=True)
    )

    if validate:
        invalid_cache = frame["cached_input_tokens"] > frame["input_tokens"]
        if invalid_cache.any():
            raise ValueError("cached_input_tokens cannot exceed input_tokens.")
        if frame["arrival_s"].isna().any():
            raise ValueError("arrival_s contains missing values after normalization.")

    return RequestTrace(
        frame=frame,
        schema=schema,
        provider=provider,
        model=model,
        region=region,
        metadata=metadata or {},
    )
