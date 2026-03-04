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

The package ships built-in Vertex AI profiles based on [Google Cloud's provisioned throughput documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput/supported-models).

### Available Models

| Model | Throughput per GSU | Output Weight | Long Context |
|-------|-------------------|---------------|--------------|
| gemini-2.0-flash-001 | 3,360 | 4x | No |
| gemini-2.0-flash-lite-001 | 6,720 | 4x | No |
| gemini-2.5-flash | 2,690 | 9x | Yes (>200k) |
| gemini-2.5-flash-lite | 8,070 | 4x | No |
| gemini-2.5-pro | 650 | 8x | Yes (>200k) |
| gemini-3.1-flash-lite-preview | 4,030 | 6x | No |

### Token Burndown Rates

Vertex AI uses different burndown rates for input vs output tokens:

- **Input tokens**: 1x weight (baseline)
- **Cached input tokens**: 0.1x weight (90% discount)
- **Output tokens**: 4-9x weight depending on model
- **Thinking tokens**: Same as output weight

### Long Context Threshold

For models with long context support, requests exceeding 200,000 input tokens use elevated weights:

- Input: 2x (instead of 1x)
- Output: 12x (instead of 8-9x)

### Usage

```python
import slosizer as slz

profile = slz.vertex_profile("gemini-2.5-flash")
```

These profiles are text-centric. If you use images, audio, video, or other token classes, add columns and extend the profile before trusting the numbers.

## Azure OpenAI PTU

Azure PTU support is calibration-first. PTU behavior is highly workload-sensitive, so we don't ship built-in profiles.

Reference: [Azure OpenAI Provisioned Throughput](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/provisioned-throughput)

### Key Characteristics

- **Workload-sensitive**: Throughput varies significantly based on prompt/completion ratios
- **Token ratio**: For GPT-4.1 and later, 1 output token ≈ 4 input tokens
- **Calibration required**: Use Azure capacity calculator + benchmarks

### Calibration Process

1. Use the [Azure capacity calculator](https://oai.azure.com/portal/calculator) to estimate baseline throughput
2. Deploy with your actual workload and measure via Azure Monitor
3. Refine the profile based on observed throughput

### Usage

```python
import slosizer as slz

profile = slz.azure_profile(
    "gpt-4.1",
    throughput_per_unit=12000.0,
    input_weight=1.0,
    output_weight=4.0,
    thinking_weight=4.0,
)
```

## Anthropic Claude

Anthropic doesn't offer a provisioned throughput model like Vertex GSU or Azure PTU. Instead, Claude uses tier-based rate limits.

Reference: [Anthropic Rate Limits](https://docs.anthropic.com/en/api/rate-limits)

### Rate Limit Tiers

| Tier | Requests/min | Input tokens/min | Output tokens/min |
|------|-------------|------------------|-------------------|
| 1 | 50 | 40,000 | 8,000 |
| 2 | 1,000 | 80,000 | 16,000 |
| 3 | 2,000 | 160,000 | 32,000 |
| 4 | 4,000 | 400,000 | 80,000 |

### Prompt Caching

Claude supports [prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) which reduces costs for repeated context:

- Cache write: 25% more expensive than base input
- Cache read: 90% cheaper than base input
- TTL: 5 minutes (extended on cache hit)

### Usage with slosizer

While slosizer is designed for provisioned throughput models, you can model Claude rate limits as a CapacityProfile:

```python
import slosizer as slz

# Model tier 3 as a single "unit"
profile = slz.CapacityProfile(
    provider="anthropic",
    model="claude-sonnet-4",
    unit_name="capacity_unit",
    throughput_per_unit=160_000 / 60,  # tokens per second
    input_weight=1.0,
    cached_input_weight=0.1,  # prompt caching discount
    output_weight=5.0,  # output tokens count more against limits
    thinking_weight=5.0,
)
```

This is a rough approximation; Claude's rate limits are more nuanced than this model captures.
