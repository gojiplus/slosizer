# slosizer: Profit-aware reserved LLM capacity planning

[![PyPI Version](https://img.shields.io/pypi/v/slosizer.svg)](https://pypi.python.org/pypi/slosizer)
[![CI](https://github.com/gojiplus/slosizer/actions/workflows/ci.yml/badge.svg)](https://github.com/gojiplus/slosizer/actions?query=workflow%3Aci)
[![Documentation Status](https://github.com/gojiplus/slosizer/actions/workflows/docs.yml/badge.svg)](https://gojiplus.github.io/slosizer/)
[![Downloads](https://static.pepy.tech/badge/slosizer)](https://pepy.tech/project/slosizer)


`slosizer` sizes reserved LLM capacity against throughput, latency SLOs, and an explicit economic objective.

It takes request traces, converts them into provider-specific capacity work, simulates queueing under bursty arrivals, and tells you how many reserved units you should buy plus how much slack capacity you are likely to carry.

The package is built for the extremely normal situation where:

- you know your request shape better than your vendor calculator does,
- you care about p95 or p99 latency, not just average throughput,
- and you do not want your capacity plan to be a sacred spreadsheet that nobody trusts.

## What problem does this solve?

Reserved-capacity systems like GSU/PTU are fundamentally **throughput constructs**, but production teams usually care about **latency SLOs**, **burst risk**, and **headroom**.

`slosizer` gives you one place to:

1. **load** request logs into the format the planner expects,
2. convert requests into provider-specific capacity units (GSU/PTU),
3. plan capacity for either:
   - **throughput**: control overload probability or required-unit percentile,
   - **latency**: satisfy p95/p99 queue-aware latency targets,
   - **hybrid**: balance provisioned cost against paygo overflow,
   - **profit**: maximize expected gross value after provisioned and SLO-failure costs,
4. quantify:
   - spare capacity,
   - overload probability,
   - expected overflow,
   - optimization benefit.

## How latency works

Total latency = **model latency** + **queue delay**.

Model latency is how long the LLM takes to process your request with no contention, estimated from token counts and provider throughput rates. Queue delay is waiting time caused by bursty arrivals: when requests arrive faster than capacity can serve them, a backlog forms.

The package simulates an FCFS queue against your request trace to estimate tail latencies (p95/p99). More reserved capacity = shorter queues = lower tail latency. The goal is finding the minimum capacity that keeps queue delay acceptable.

## Two ways to start

### Option 1: No data yet

Use the synthetic generator to explore capacity planning before you have real logs:

```python
import slosizer as slz

trace = slz.make_synthetic_trace(seed=42)
profile = slz.vertex_profile("gemini-2.5-flash-lite")

result = slz.plan_capacity(
    trace,
    profile,
    slz.LatencyTarget(slz.LatencySLO(threshold_s=1.5, percentile=0.99, metric="e2e")),
)
```

### Option 2: You have request logs

You need a CSV (or DataFrame) with at minimum these 3 columns:

| Column | What it means |
|--------|---------------|
| `timestamp` | When the request arrived (datetime or seconds) |
| `input_tokens` | Tokens in the prompt |
| `output_tokens` | Tokens in the response |

That's it. The package normalizes timestamps and fills defaults for everything else.

```python
import pandas as pd
import slosizer as slz

df = pd.read_csv("requests.csv")

trace = slz.from_dataframe(
    df,
    schema=slz.RequestSchema(
        time_col="timestamp",
        input_tokens_col="input_tokens",
        output_tokens_col="output_tokens",
    ),
    provider="vertex",
    model="gemini-2.5-flash-lite",
)
```

## Quickstart

### 1) Create the environment with `uv`

```bash
uv sync --all-groups
```

### 2) Run the shipped synthetic demo

```bash
uv run python examples/quickstart.py
```

This writes:

- `examples/output/comparison.csv`
- `examples/output/latency_vs_capacity.png`
- `examples/output/required_units_distribution.png`
- `examples/output/scenario_benefit.png`
- `examples/output/percentile_tradeoff.png`

### 3) Run the checks

```bash
uv run pytest -q
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run deptry .
uv run vulture
```

## Install and use it on your own trace

### Minimal latency-oriented example

```python
import pandas as pd
import slosizer as slz

df = pd.read_csv("requests.csv")

trace = slz.from_dataframe(
    df,
    schema=slz.RequestSchema(
        time_col="timestamp",
        class_col="route",
        input_tokens_col="prompt_tokens",
        cached_input_tokens_col="cached_prompt_tokens",
        output_tokens_col="completion_tokens",
        thinking_tokens_col="reasoning_tokens",
        max_output_tokens_col="max_output_tokens",
        latency_col="latency_s",
    ),
    provider="vertex",
    model="gemini-2.5-flash-lite",
)

profile = slz.vertex_profile("gemini-2.5-flash-lite")

result = slz.plan_capacity(
    trace,
    profile,
    slz.LatencyTarget(
        slz.LatencySLO(
            threshold_s=1.5,
            percentile=0.99,
            metric="e2e",
        )
    ),
)

print(result.recommended_units)
print(result.metrics)
```

### Throughput-oriented example

```python
import slosizer as slz

trace = slz.make_synthetic_trace(seed=42)
profile = slz.vertex_profile("gemini-2.5-flash-lite")

result = slz.plan_capacity(
    trace,
    profile,
    slz.ThroughputTarget(
        percentile=0.99,
        max_overload_probability=0.01,
        windows_s=(1.0, 5.0, 30.0),
    ),
)

print(result.recommended_units)
print(result.slack_summary)
```

### Cost-optimal hybrid planning

This is the normal operating case. The request trace is the demand forecast. With fixed demand and a hard SLO, minimizing inference cost subject to the SLO maximizes profit. Users do not need to estimate request value or demand elasticity.

```python
from datetime import date

import slosizer as slz

trace = slz.make_synthetic_trace(seed=42)
profile = slz.vertex_profile("gemini-2.5-flash")
pricing = slz.RateCard(
    provisioned=slz.ProvisionedPricing(cost_per_unit_hour=3.698630137),
    paygo=slz.PaygoPricing(
        input_cost_per_million=0.30,
        cached_input_cost_per_million=0.03,
        output_cost_per_million=2.50,
        thinking_cost_per_million=2.50,
    ),
    currency="USD",
    provider="vertex",
    model="gemini-2.5-flash",
    verified_on=date(2026, 8, 15),
    source="https://cloud.google.com/vertex-ai/generative-ai/pricing",
)

result = slz.plan_hybrid_capacity(
    trace,
    profile,
    pricing,
    slz.HybridTarget(
        strategy="cost_optimal",
        latency_slo=slz.LatencySLO(
            threshold_s=1.5,
            percentile=0.99,
        ),
    ),
    options=slz.PlanOptions(baseline_latency_model=slz.BaselineLatencyModel()),
)

print(f"Provision {result.provisioned_units} GSUs + paygo overflow")
print(
    f"Saves ${result.savings_vs_full_provision:.2f}/hr ({result.savings_percent:.0f}%)"
)
```

That public list rate was checked on 2026-08-15. Production analysis should use the effective rate on your invoice or contract and record its source and validity dates.

`cost_optimal` finds the cheapest provisioned and paygo blend. `percentile_split` provisions at a chosen workload percentile and sends the rest to paygo.

### Advanced economic planning

Use `plan_profit_capacity()` when you need an absolute profit estimate or want to price SLO misses instead of treating the SLO as a hard constraint. This requires expected gross value per request and, for a priced SLO, a defensible cost per miss.

```python
import slosizer as slz

trace = slz.make_synthetic_trace(seed=42)
profile = slz.vertex_profile("gemini-2.5-flash")
pricing = slz.RateCard(
    provisioned=slz.ProvisionedPricing(cost_per_unit_hour=3.698630137),
    provider="vertex",
    model="gemini-2.5-flash",
)

result = slz.plan_profit_capacity(
    trace,
    profile,
    pricing,
    slz.ProfitTarget(
        latency_slo=slz.LatencySLO(threshold_s=1.5, percentile=0.99),
        slo_policy="priced",
        value_per_request=0.05,
        slo_violation_cost_per_request=0.01,
    ),
    options=slz.PlanOptions(baseline_latency_model=slz.BaselineLatencyModel()),
)

print(result.expected_profit_hourly)
print(result.candidate_plans)
```

`business_value` is gross contribution before inference and SLO costs. It changes the value assigned to a request, not demand. The package does not estimate demand effects. If a model is expected to receive different traffic, supply a different forecast trace in an optional `ProfitScenario`. Most users should not need this API.

With one trace and a hard SLO, business value changes reported profit but not recommended capacity. Use the hybrid planner for that case.

`headroom_factor` is rejected for `cost_optimal` because adding capacity after the search would no longer be cost optimal. Model uncertainty with workload scenarios instead.

### Azure PTU example

Azure support is calibration-first: you seed a profile from the Azure calculator and benchmark results, then use the same planning machinery.

```python
import slosizer as slz

profile = slz.azure_profile(
    "gpt-5.2",
    throughput_per_unit=3400 / 60,
    purchase_increment=5,
    min_units=15,
    input_weight=1.0,
    cached_input_weight=0.0,
    output_weight=8.0,
    thinking_weight=8.0,
    deployment_type="data_zone_provisioned",
)
```

## Optional fields for better planning

The 3-column minimum works, but you get more accurate capacity estimates with:

| Column | Why it helps |
|--------|--------------|
| `cached_input_tokens` | Cached tokens cost less capacity |
| `thinking_tokens` | Reasoning models use extra tokens |
| `max_output_tokens` | Helps estimate worst-case latency |
| `class_name` | Separate capacity needs by request type |
| `latency_s` | Calibrate model latency estimates |
| `request_model` / `response_model` | Distinguish the requested alias from the model that actually served |
| `service_tier` | Separate provisioned, standard, priority, batch, and other routes |
| `business_value` | Optional absolute-profit or priced-SLO analysis |

See [`docs/data-requirements.md`](https://gojiplus.github.io/slosizer/data-requirements.html) for full details.

Example input files:

- [`examples/input/synthetic_request_trace_baseline.csv`](https://github.com/gojiplus/slosizer/blob/main/examples/input/synthetic_request_trace_baseline.csv)
- [`examples/input/synthetic_request_trace_optimized.csv`](https://github.com/gojiplus/slosizer/blob/main/examples/input/synthetic_request_trace_optimized.csv)

## Built-in provider support

### Vertex GSU
The package ships a reviewed, versioned TOML catalog for current text-capable Vertex Provisioned Throughput models, including:

- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`
- `gemini-2.5-pro`
- `gemini-3.1-flash-lite`
- `gemini-3.1-pro-preview`
- `gemini-3.5-flash`
- `gemini-3.5-flash-lite`
- `gemini-3.6-flash`
- `gemini-3.7-flash`

Provider facts live in `src/slosizer/data/vertex.toml`, not in optimizer code. Add a new model by updating a catalog or load your own with `load_capacity_profiles()`.

### Azure PTU
Azure PTU support is user-calibrated on purpose. The package gives you the same planning engine, but you provide the model-specific PTU profile from your calculator + benchmark loop.

See [`docs/provider-adapters.md`](https://gojiplus.github.io/slosizer/provider-adapters.html).

## Synthetic demo: what it shows

The repo ships with a fake but bursty workload containing three classes:

- chat
- rag
- reasoning

The optimized variant simulates:

- tighter prompts,
- more caching,
- shorter outputs,
- lower thinking-token budgets.

That lets you inspect two things immediately:

1. **Optimization can reduce reserved-capacity needs.**
2. **Planning for stricter percentiles usually increases slack capacity.**

### Snapshot of the current synthetic outputs

| scenario | objective | target | recommended units | avg spare fraction (1s) | overload probability (1s) | achieved latency quantile |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| baseline | latency | p95 <= 1.5s | 2 | 0.718 | 0.031 | 1.320s |
| baseline | latency | p99 <= 1.5s | 3 | 0.807 | 0.004 | 1.413s |
| baseline | throughput | p99 units, overload <= 1% | 3 | 0.807 | 0.004 | - |
| optimized | latency | p95 <= 1.5s | 2 | 0.779 | 0.009 | 0.954s |
| optimized | latency | p99 <= 1.5s | 2 | 0.779 | 0.009 | 1.251s |
| optimized | throughput | p99 units, overload <= 1% | 2 | 0.779 | 0.009 | - |

These numbers are synthetic. They are there to show the mechanics, not to cosplay as your production traffic.

### Output plots

**Latency vs provisioned capacity**

![Latency vs capacity](https://raw.githubusercontent.com/gojiplus/slosizer/main/docs/assets/latency_vs_capacity.png)

**Distribution of required reserved units**

![Required units distribution](https://raw.githubusercontent.com/gojiplus/slosizer/main/docs/assets/required_units_distribution.png)

**Optimization benefit**

![Optimization benefit](https://raw.githubusercontent.com/gojiplus/slosizer/main/docs/assets/scenario_benefit.png)

**Slack trade-off**

![Slack trade-off](https://raw.githubusercontent.com/gojiplus/slosizer/main/docs/assets/percentile_tradeoff.png)

## Repo map

- [`docs/formalization.md`](https://gojiplus.github.io/slosizer/formalization.html): generic throughput/latency model
- [`docs/economics-and-data.md`](https://gojiplus.github.io/slosizer/economics-and-data.html): profit objective, SLO policy, and storage boundaries
- [`docs/data-requirements.md`](https://gojiplus.github.io/slosizer/data-requirements.html): what columns you need and why
- [`docs/provider-adapters.md`](https://gojiplus.github.io/slosizer/provider-adapters.html): how GSU/PTU adaptation works
- [`docs/examples.md`](https://gojiplus.github.io/slosizer/examples.html): the synthetic walkthrough
- [`examples/quickstart.py`](https://github.com/gojiplus/slosizer/blob/main/examples/quickstart.py): reproducible demo script

## Caveats

- The queue model is intentionally simple: FCFS fluid queueing, not a perfect service simulator.
- Built-in Vertex profiles are text-centric. Multimodal traffic needs more columns and weights.
- Azure PTU math is workload-sensitive, so the package does not fake vendor-authoritative PTU values for you.
- If you do not have a latency column, the package falls back to a simple token-based baseline latency model. That is a starting point, not gospel.

## Name

The package name is **`slosizer`** because "how many units do I need, and how much empty air am I buying to hit p99?" is the real question under all the vendor jargon.
