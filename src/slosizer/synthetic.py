"""Synthetic workload generation for testing and demonstration."""

import numpy as np
import pandas as pd

from slosizer.ingest import from_dataframe
from slosizer.schema import RequestSchema, RequestTrace


def _base_trace(seed: int = 42, horizon_s: int = 4 * 3600) -> pd.DataFrame:
    """Generate a realistic synthetic request trace.

    Creates a 4-hour trace with diurnal patterns, burst events, and
    three request classes (chat, rag, reasoning).

    Args:
        seed: Random seed for reproducibility.
        horizon_s: Trace duration in seconds.

    Returns:
        DataFrame with synthetic request data.
    """
    rng = np.random.default_rng(seed)
    secs = np.arange(horizon_s)

    lam = 1.5 + 0.5 * np.sin(2 * np.pi * secs / 3600 - 0.8) + 0.3 * np.sin(2 * np.pi * secs / 900)
    lam = np.clip(lam, 0.2, None)
    for _ in range(14):
        start = rng.integers(0, horizon_s - 200)
        dur = int(rng.integers(20, 160))
        amp = rng.uniform(1.5, 4.5)
        decay = np.exp(-np.linspace(0, 2, dur))
        lam[start : start + dur] += amp * decay

    counts = rng.poisson(lam)
    n = int(counts.sum())
    sec_rep = np.repeat(secs, counts)
    arrivals = sec_rep + rng.random(n)
    classes = rng.choice(["chat", "rag", "reasoning"], p=[0.55, 0.25, 0.20], size=n)

    input_tokens = np.empty(n, dtype=int)
    output_tokens = np.empty(n, dtype=int)
    thinking_tokens = np.zeros(n, dtype=int)
    cached_frac = np.zeros(n)
    max_output_tokens = np.empty(n, dtype=int)

    for class_name in ["chat", "rag", "reasoning"]:
        idx = np.where(classes == class_name)[0]
        size = len(idx)
        if class_name == "chat":
            inp = rng.lognormal(mean=np.log(700), sigma=0.45, size=size)
            out = rng.gamma(shape=3.5, scale=55, size=size) + 30
            max_out = np.full(size, 512)
            hit = rng.random(size) < 0.18
            cache = np.where(hit, rng.uniform(0.20, 0.50, size=size), 0.0)
            think = np.zeros(size)
        elif class_name == "rag":
            inp = rng.lognormal(mean=np.log(2200), sigma=0.55, size=size)
            out = rng.gamma(shape=4.0, scale=65, size=size) + 50
            max_out = np.full(size, 768)
            hit = rng.random(size) < 0.30
            cache = np.where(hit, rng.uniform(0.25, 0.65, size=size), 0.0)
            think = np.zeros(size)
        else:
            inp = rng.lognormal(mean=np.log(1700), sigma=0.60, size=size)
            out = rng.gamma(shape=4.5, scale=85, size=size) + 80
            max_out = np.full(size, 1024)
            hit = rng.random(size) < 0.10
            cache = np.where(hit, rng.uniform(0.05, 0.25, size=size), 0.0)
            think = rng.gamma(shape=3.0, scale=140, size=size)

        input_tokens[idx] = np.round(inp).astype(int)
        output_tokens[idx] = np.minimum(np.round(out).astype(int), max_out)
        thinking_tokens[idx] = np.round(think).astype(int)
        cached_frac[idx] = cache
        max_output_tokens[idx] = max_out

    cached_input_tokens = np.round(input_tokens * cached_frac).astype(int)

    return pd.DataFrame(
        {
            "ts": arrivals,
            "class_name": classes,
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "max_output_tokens": max_output_tokens,
        }
    )


def optimize_trace(trace: RequestTrace) -> RequestTrace:
    """Apply prompt optimization to reduce token usage.

    Simulates the effect of prompt engineering and caching improvements
    by reducing input, output, and thinking tokens.

    Args:
        trace: Original request trace.

    Returns:
        Optimized trace with reduced token counts.
    """
    frame = trace.frame.copy()

    extra_cache = np.select(
        [frame.class_name == "chat", frame.class_name == "rag", frame.class_name == "reasoning"],
        [0.12, 0.18, 0.06],
        default=0.0,
    )
    reduce_input = np.select(
        [frame.class_name == "chat", frame.class_name == "rag", frame.class_name == "reasoning"],
        [0.12, 0.22, 0.10],
        default=0.0,
    )
    reduce_output = np.select(
        [frame.class_name == "chat", frame.class_name == "rag", frame.class_name == "reasoning"],
        [0.20, 0.18, 0.12],
        default=0.0,
    )
    reduce_thinking = np.select([frame.class_name == "reasoning"], [0.35], default=0.0)

    frame["input_tokens"] = np.round(frame["input_tokens"] * (1 - reduce_input)).astype(int)
    original_cache = np.divide(
        trace.frame["cached_input_tokens"],
        np.maximum(trace.frame["input_tokens"], 1),
    )
    new_cache = np.clip(original_cache + extra_cache, 0, 0.8)
    frame["cached_input_tokens"] = np.round(frame["input_tokens"] * new_cache).astype(int)
    frame["output_tokens"] = np.minimum(
        np.round(frame["output_tokens"] * (1 - reduce_output)).astype(int),
        np.round(frame["max_output_tokens"] * 0.85).astype(int),
    )
    frame["thinking_tokens"] = np.round(frame["thinking_tokens"] * (1 - reduce_thinking)).astype(
        int
    )
    frame["max_output_tokens"] = np.round(frame["max_output_tokens"] * 0.85).astype(int)

    raw = frame.rename(columns={"arrival_s": "ts"})
    return from_dataframe(
        raw,
        schema=RequestSchema(
            time_col="ts",
            class_col="class_name",
            input_tokens_col="input_tokens",
            cached_input_tokens_col="cached_input_tokens",
            output_tokens_col="output_tokens",
            thinking_tokens_col="thinking_tokens",
            max_output_tokens_col="max_output_tokens",
        ),
        provider=trace.provider,
        model=trace.model,
        region=trace.region,
        metadata={**trace.metadata, "optimized_from": trace.metadata.get("scenario", "baseline")},
    )


def make_synthetic_trace(
    *,
    horizon_s: int = 4 * 3600,
    seed: int = 42,
    scenario: str = "baseline",
) -> RequestTrace:
    """Generate a synthetic request trace for testing.

    Args:
        horizon_s: Trace duration in seconds.
        seed: Random seed for reproducibility.
        scenario: Either "baseline" or "optimized".

    Returns:
        Synthetic RequestTrace.
    """
    raw = _base_trace(seed=seed, horizon_s=horizon_s)
    trace = from_dataframe(
        raw,
        schema=RequestSchema(
            time_col="ts",
            class_col="class_name",
            input_tokens_col="input_tokens",
            cached_input_tokens_col="cached_input_tokens",
            output_tokens_col="output_tokens",
            thinking_tokens_col="thinking_tokens",
            max_output_tokens_col="max_output_tokens",
        ),
        metadata={"scenario": scenario, "seed": seed},
    )
    if scenario == "optimized":
        return optimize_trace(trace)
    return trace
