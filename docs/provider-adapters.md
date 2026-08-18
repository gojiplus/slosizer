# Provider Adapters

The package is generic in the middle and provider-specific at the edges.

## CapacityProfile

A provider adapter boils down to a `CapacityProfile`:

- `throughput_per_unit`
- `purchase_increment`
- `min_units`
- `input_weight`
- `cached_input_weight`
- `output_weight`
- `thinking_weight`
- optional long-context overrides

That is enough to turn requests into adjusted work and then into required reserved units.

## Vertex AI GSU

The package loads reviewed Vertex AI profiles from `src/slosizer/data/vertex.toml`. The catalog records [Google Cloud's Provisioned Throughput documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput/supported-models) as its source and includes the date the values were checked.

### Available models

Use `available_vertex_profiles()` to read the current model set from the
catalog. Keeping the model identifiers and capacity values in one runtime
source prevents documentation from drifting from planner behavior.

### Token Burndown Rates

Vertex AI uses different burndown rates for input vs output tokens:

- **Input tokens**: 1x weight (baseline)
- **Cached input tokens**: 0.1x weight (90% discount)
- **Output tokens**: 4-9x weight depending on model
- **Thinking tokens**: Same as output weight

### Long Context Threshold

For cataloged models with a long-context rule, requests exceeding 200,000 input tokens use the elevated weights stated by the provider. The exact output weight depends on the model.

- Input: 2x (instead of 1x)
- Output: 9x or 12x

### Usage

```python
import slosizer as slz

profile = slz.vertex_profile("gemini-2.5-flash")
```

These profiles are text-centric. If you use images, audio, video, or other token classes, add columns and extend the profile before trusting the numbers.

To keep a private or faster-moving catalog outside the package:

```python
import slosizer as slz

profiles = slz.load_capacity_profiles("capacity-profiles.toml")
profile = profiles["my-exact-model-id"]
```

## Azure OpenAI PTU

Azure PTU support is calibration-first. PTU behavior is highly workload-sensitive, so we don't ship built-in profiles.

Reference: [Microsoft Foundry PTU sizing](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/provisioned-throughput-sizing)

### Key Characteristics

- Throughput depends on the exact model, version, deployment type, prompt size, response size, cache rate, and call rate.
- Current models publish input TPM per PTU and a model-specific output-to-input ratio.
- Cached input does not consume PTU capacity for models covered by the current sizing formula.
- The published values are estimates. Use representative benchmarks and the Provisioned-managed Utilization V2 metric before buying a reservation.

### Calibration Process

1. Use the Foundry sizing table or capacity calculator to estimate baseline throughput
2. Deploy with your actual workload and measure via Azure Monitor
3. Refine the profile based on observed throughput

### Usage

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

`throughput_per_unit` is adjusted tokens per second, so divide the provider's input TPM per PTU by 60. The example values come from the current GPT-5.2 Data Zone sizing example in the official documentation; check the table again before use.

## Anthropic Claude (Planned)

> **Status: Not Yet Implemented**
>
> Anthropic doesn't offer a provisioned throughput model like Vertex GSU or Azure PTU. Claude uses tier-based rate limits which don't map cleanly to slosizer's capacity unit model. A future version may add support for modeling Claude rate limits, but there is currently no built-in `anthropic_profile()` function.

Reference: [Anthropic Rate Limits](https://docs.anthropic.com/en/api/rate-limits)
