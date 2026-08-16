# Economics and data architecture

For fixed demand and a hard SLO, minimizing inference cost subject to the SLO maximizes profit. `plan_hybrid_capacity()` is the default API for that case. The request trace is the demand forecast, so users do not need to supply request value or a demand model.

Keep usage facts, provider capacity facts, prices, and business policy in separate stores. Each changes on a different schedule and comes from a different authority. Combining them in one model table makes historical plans impossible to reproduce and new models risky to add.

| Layer | What belongs there | Where to keep it |
| --- | --- | --- |
| Usage facts | Request identity, requested and served model, service tier, token counts, timing, outcome, and business value | An append-only telemetry or warehouse table; a `RequestTrace` is the planner's in-memory view |
| Capacity catalog | Throughput per unit, purchase constraints, burndown weights, deployment type, source, and verification date | Reviewed TOML in source control or an external catalog loaded with `load_capacity_profiles()` |
| Rate cards | Currency, provider, model, region, service tier, effective dates, provisioned price, and token prices | A private effective-dated table tied to contracts and invoices |
| Policy | SLO, hard or priced treatment, business value, forecast scenario, and search bounds | Application configuration passed to pure planning functions |

The planner joins these layers for one decision. It should not rewrite any of them.

## Usage facts

Record what the application requested and what the provider returned. These can differ when an alias, gateway, fallback, or provider migration selects the serving model. The canonical trace supports:

- `request_id`
- `request_model`
- `response_model`
- `service_tier`
- input, cached input, output, and separately reported thinking tokens
- observed latency
- request class
- `business_value`

These names follow the distinctions in the [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai), including requested model, response model, cache reads, and reasoning tokens. Store provider request IDs and status fields in the warehouse even if the capacity planner does not yet use them. Do not put prompts or responses in the planning table. They add privacy risk without helping capacity arithmetic.

`business_value` is optional. `plan_profit_capacity()` uses it to report absolute profit or price SLO misses. It is expected gross contribution before inference and SLO costs. It changes the value assigned to a request, not the number or timing of requests.

## Capacity catalog

Provider capacity facts belong in a reviewed catalog because they change when models and deployment products change. The built-in Vertex catalog lives in `src/slosizer/data/vertex.toml`. Every entry uses an exact model ID and records the official source and the date it was checked.

Do not fetch this data during planning. Runtime fetching makes the same input produce different answers on different days. Update the catalog in a normal code review, test the changed profiles, and preserve the old rate card used by past decisions.

Azure profiles remain calibration based. Current Foundry guidance says capacity depends on model, model version, prompt size, response size, cache rate, and call rate. It recommends the sizing tables or calculator followed by a benchmark on representative traffic. Store that measured calibration as a profile with its deployment type and region ([Microsoft Foundry PTU sizing](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/provisioned-throughput-sizing)).

## Rate cards

Public list prices are useful for exploration. Profit decisions should use the effective cost from the contract or invoice. A rate card needs at least:

- provider and exact model
- deployment type, service tier, and region
- currency
- first and last effective date
- source verification date
- provisioned cost per unit-hour
- uncached input, cached input, output, and reasoning-token prices
- source or contract reference

`RateCard.validate_for()` rejects a rate card explicitly scoped to a different provider, model, region, or deployment type. The hybrid planner charges cached and thinking tokens separately. If either token price is absent, it uses the regular input or output price, which is conservative.

Keep invoice cost separate from request usage. Provider usage APIs can lag or disagree with invoice totals, and [OpenAI's Usage API documentation](https://platform.openai.com/docs/api-reference/usage) tells users to use its Costs endpoint for financial reconciliation. The [FOCUS 1.4 specification](https://focus.finops.org/focus-specification/) supplies a useful normalized shape for cost, usage, and contract commitments across providers.

## Decision rule

The common case has a fixed trace and a hard SLO. Gross value is then constant across capacity choices, so it drops out of the choice. `plan_hybrid_capacity()` minimizes the cost of provisioned capacity and paygo overflow among plans that satisfy the SLO. This is profit maximization without a business-value input.

For model scenario `m` and provisioned units `u`, the reserved-capacity planner computes:

`profit(m, u) = gross_value(m) - provisioned_cost(m, u) - SLO_cost(m, u)`

`plan_profit_capacity()` chooses the legal `u` with the highest expected hourly profit and returns every candidate. Use it when the analysis needs an absolute profit estimate or a priced SLO. It plans reserved capacity; use `plan_hybrid_capacity()` for the normal provisioned and paygo blend.

For one trace and a hard SLO, business value changes reported profit but not the recommended capacity. The hard SLO fixes the eligible choices and gross value is identical for all of them.

`compare_profit_scenarios()` is an optional cross-model analysis. It ranks model scenarios after optimizing reserved capacity within each scenario.

Do not use `headroom_factor` with profit optimization. Adding capacity after finding the optimum changes the chosen objective. Represent forecast uncertainty with separate workload traces and compare their results.

## Demand assumptions

The package does not estimate demand effects. Each `RequestTrace` supplies the forecast request count, timing, token mix, and burstiness. The same trace across alternatives means demand is fixed.

If a model is expected to receive different traffic, create a separate forecast trace and pass it through `ProfitScenario`. This advanced case should rest on an experiment or forecast. The API does not expose a generic demand multiplier because a traffic change can alter arrival patterns and request composition as well as total volume.

## SLO policy

An SLO defines the share of requests that must meet a latency threshold. A p99 SLO allows one percent of eligible requests to miss the threshold. That allowance is the request-based error budget described in [Google Cloud's SLO guidance](https://docs.cloud.google.com/stackdriver/docs/solutions/slo-monitoring).

`ProfitTarget` supports two policies:

- `hard` removes any capacity choice whose observed attainment is below the target percentile. Use it for a product promise, safety boundary, or contract that the planner may not trade away.
- `priced` keeps all capacity choices and subtracts a business cost for each request over the threshold. Use it only when the cost is estimated from evidence.

A hard SLO still represents an economic choice. The organization chose the threshold and error budget because the expected cost of worse service exceeds the capacity savings. The optimizer respects that prior choice instead of inventing a failure price.

Hybrid latency checks are conservative. Without a measured latency model for the paygo route, the planner does not assume that all-paygo traffic satisfies the SLO. Azure notes that spillover can add latency even though it reduces disruption during bursts ([Microsoft Foundry spillover](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/spillover-traffic-management)).

## Adding a model

Add a model without changing planner logic:

1. Create or update a capacity catalog entry from official provider documentation or a benchmark.
2. Record the source, verification date, exact model ID, deployment type, and region.
3. Add an effective-dated rate card from the contract, invoice, or current list price.
4. Replay representative usage and run fixed-demand hybrid planning under the product SLO.
5. Add model-specific business value or demand forecasts only for a supported cross-model profit comparison.
6. Retain the chosen plan and candidate frontier with the decision record.

This process keeps a provider announcement from silently changing an old plan and makes each assumption replaceable when better evidence arrives.
