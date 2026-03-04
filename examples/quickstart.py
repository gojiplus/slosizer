from __future__ import annotations

from pathlib import Path

import slosizer as slz

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

profile = slz.vertex_profile("gemini-2.0-flash-001")
baseline = slz.make_synthetic_trace(seed=42, scenario="baseline")
optimized = slz.make_synthetic_trace(seed=42, scenario="optimized")

targets = [
    slz.LatencyTarget(slz.LatencySLO(threshold_s=1.5, percentile=0.95)),
    slz.LatencyTarget(slz.LatencySLO(threshold_s=1.5, percentile=0.99)),
    slz.ThroughputTarget(
        percentile=0.99, max_overload_probability=0.01, windows_s=(1.0, 5.0, 30.0)
    ),
]

comparison = slz.compare_scenarios(
    {"baseline": baseline, "optimized": optimized},
    profile,
    targets,
)
comparison.to_csv(OUTPUT_DIR / "comparison.csv", index=False)

baseline.frame.to_csv(OUTPUT_DIR / "synthetic_request_trace_baseline.csv", index=False)
optimized.frame.to_csv(OUTPUT_DIR / "synthetic_request_trace_optimized.csv", index=False)

latency_target = slz.LatencyTarget(slz.LatencySLO(threshold_s=1.5, percentile=0.99))
slz.plot_latency_vs_units(
    baseline,
    profile,
    units=range(1, 13),
    target=latency_target,
    path=OUTPUT_DIR / "latency_vs_capacity.png",
)
slz.plot_required_units_distribution(
    baseline,
    profile,
    path=OUTPUT_DIR / "required_units_distribution.png",
)
slz.plot_capacity_tradeoff(
    comparison,
    path=OUTPUT_DIR / "scenario_benefit.png",
)
slz.plot_slack_tradeoff(
    comparison,
    path=OUTPUT_DIR / "percentile_tradeoff.png",
)

print("Wrote synthetic example outputs to", OUTPUT_DIR)
print(comparison.round(3).to_string(index=False))
