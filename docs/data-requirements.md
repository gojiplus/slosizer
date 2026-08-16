# Data requirements

You can start with only three fields:

| Field | Required | Meaning |
| --- | --- | --- |
| timestamp | yes | request arrival time; numeric seconds or a datetime-like column |
| input_tokens | yes | all prompt tokens, including the cached subset |
| output_tokens | yes | response tokens excluding separately reported thinking tokens |

The package gets more useful when you also provide these:

| Field | Recommended | Why it matters |
| --- | --- | --- |
| cached_input_tokens | yes | some providers discount cached tokens |
| thinking_tokens | yes | reasoning-heavy routes can burn far more reserved capacity |
| max_output_tokens | yes | useful for conservative planning and admission-style estimates |
| class_name | yes | lets you segment chat, RAG, tool use, reasoning, and other traffic classes |
| latency_s | yes | helps fit a baseline-latency model from real telemetry |
| request_id | yes | joins the planning row to traces, outcomes, and billing evidence |
| request_model | yes | records the model or alias requested by the application |
| response_model | yes | records the exact model that served the request |
| service_tier | yes | separates provisioned, standard, priority, batch, and fallback traffic |
| business_value | advanced profit planning only | expected gross contribution before inference and SLO costs; does not change forecast demand |

## Minimal schema

```text
timestamp,input_tokens,output_tokens
0.0,800,120
0.2,1200,180
0.7,600,95
```

## Recommended schema

```text
timestamp,request_id,class_name,request_model,response_model,service_tier,input_tokens,cached_input_tokens,output_tokens,thinking_tokens,max_output_tokens,latency_s,business_value
0.0,r1,chat,gemini-2.5-flash,gemini-2.5-flash,provisioned,800,120,120,0,512,0.74,0.04
0.2,r2,rag,gemini-2.5-flash,gemini-2.5-flash,provisioned,1200,350,180,0,768,1.10,0.08
0.7,r3,reasoning,gemini-2.5-flash,gemini-2.5-flash,standard,600,0,95,210,1024,1.48,0.15
```

The requested and response model fields are separate on purpose. A gateway, alias, fallback, or provider migration can make them differ. Cached and thinking tokens stay separate because capacity burndown and paygo prices can differ from ordinary input and output tokens.

The canonical token groups do not overlap. `cached_input_tokens` is a subset of `input_tokens`, so the planner subtracts the cached subset before applying the ordinary input weight. `thinking_tokens` is additional to `output_tokens`. OpenTelemetry defines reasoning output as a subset of total output. When importing that shape, set canonical `output_tokens` to total output minus reasoning output. Otherwise the planner will count reasoning twice.

Keep provider billing exports in a cost table rather than copying invoice cost onto every request. Join usage to an effective-dated rate card for planning, then reconcile aggregate cost to the invoice.

## Example files in this repo

- `examples/input/synthetic_request_trace_baseline.csv`
- `examples/input/synthetic_request_trace_optimized.csv`

Those are fake, but structurally realistic.
