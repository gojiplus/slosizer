# Data requirements

You can start with only three fields:

| Field | Required | Meaning |
| --- | --- | --- |
| timestamp | yes | request arrival time; numeric seconds or a datetime-like column |
| input_tokens | yes | request input or prompt tokens |
| output_tokens | yes | generated response tokens |

The package gets more useful when you also provide these:

| Field | Recommended | Why it matters |
| --- | --- | --- |
| cached_input_tokens | yes | some providers discount cached tokens |
| thinking_tokens | yes | reasoning-heavy routes can burn far more reserved capacity |
| max_output_tokens | yes | useful for conservative planning and admission-style estimates |
| class_name | yes | lets you segment chat, RAG, tool use, reasoning, and other traffic classes |
| latency_s | yes | helps fit a baseline-latency model from real telemetry |

## Minimal schema

```csv
timestamp,input_tokens,output_tokens
0.0,800,120
0.2,1200,180
0.7,600,95
```

## Recommended schema

```csv
timestamp,class_name,input_tokens,cached_input_tokens,output_tokens,thinking_tokens,max_output_tokens,latency_s
0.0,chat,800,120,120,0,512,0.74
0.2,rag,1200,350,180,0,768,1.10
0.7,reasoning,600,0,95,210,1024,1.48
```

## Example files in this repo

- `examples/input/synthetic_request_trace_baseline.csv`
- `examples/input/synthetic_request_trace_optimized.csv`

Those are fake, but structurally realistic.
