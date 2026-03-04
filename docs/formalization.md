# Formalization

`slosizer` treats reserved-capacity planning as two related problems.

1. **Throughput planning**  
   Convert every request into provider-specific capacity work, then ask how many reserved units are needed so burst windows stay inside budget.

2. **Latency planning**  
   Split end-to-end latency into baseline model latency plus queue delay induced by bursty arrivals and finite reserved capacity.

## Generic request representation

Each request has:

- arrival time `t`
- input tokens `I`
- cached input tokens `C`
- output tokens `O`
- thinking tokens `H`

A provider profile supplies the burndown weights:

- `w_in`
- `w_cache`
- `w_out`
- `w_think`

The request work is:

`B = w_in * I + w_cache * C + w_out * O + w_think * H`

If a profile has long-context rules, the weights can change once the input-token threshold is crossed.

## Throughput planning

For a window of length `Delta`, the total work in the bucket is:

`D_n(Delta) = sum(B_j for requests in window n)`

If one reserved unit serves `tau` adjusted tokens per second, the required reserved units in that bucket are:

`X_n(Delta) = D_n(Delta) / (tau * Delta)`

That lets us compute:

- mean required units
- p95 / p99 required units
- overload probability `P(X_n > G)`
- expected overflow `E[(X_n - G)+]`
- average spare capacity `E[(G - X_n)+]`

## Latency planning

Reserved capacity affects latency through queueing, not through the intrinsic model floor.

`R = L_base + W`

Where:

- `L_base`: model latency with no capacity contention
- `W`: queue delay caused by backlog

The package uses a simple FCFS fluid queue:

`Q_(n+1) = max(0, Q_n + arrivals_work - service_rate * elapsed_time)`

Queue delay for request `j` is approximated by:

`W_j = backlog_before_j / service_rate`

This is not a perfect service simulator; it is a deliberately pragmatic tail-latency approximation.

## Why percentile choice matters

Optimizing for p95 usually buys fewer reserved units and therefore lower average slack.  
Optimizing for p99 buys more headroom and therefore lower overload probability, but also more idle capacity on average.

That trade-off is not a bug. It is the whole game.
