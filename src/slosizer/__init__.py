"""Capacity planning for reserved LLM throughput.

slosizer helps you determine how much reserved capacity to purchase from
cloud LLM providers (Vertex AI GSU, Azure PTU) based on your workload
characteristics and SLO targets.
"""

from slosizer.ingest import from_dataframe
from slosizer.planning import compare_scenarios, plan_capacity
from slosizer.plotting import (
    plot_capacity_tradeoff,
    plot_latency_vs_units,
    plot_required_units_distribution,
    plot_slack_tradeoff,
)
from slosizer.providers.azure import azure_profile
from slosizer.providers.vertex import available_vertex_profiles, vertex_profile
from slosizer.schema import (
    BaselineLatencyModel,
    CapacityProfile,
    LatencyMetric,
    LatencySLO,
    LatencyTarget,
    OutputTokenSource,
    PlanOptions,
    PlanResult,
    RequestSchema,
    RequestTrace,
    SimulationResult,
    ThroughputTarget,
)
from slosizer.simulation import fit_baseline_latency_model, simulate_capacity
from slosizer.synthetic import make_synthetic_trace, optimize_trace

__all__ = [
    "BaselineLatencyModel",
    "CapacityProfile",
    "LatencyMetric",
    "LatencySLO",
    "LatencyTarget",
    "OutputTokenSource",
    "PlanOptions",
    "PlanResult",
    "RequestSchema",
    "RequestTrace",
    "SimulationResult",
    "ThroughputTarget",
    "available_vertex_profiles",
    "azure_profile",
    "compare_scenarios",
    "fit_baseline_latency_model",
    "from_dataframe",
    "make_synthetic_trace",
    "optimize_trace",
    "plan_capacity",
    "plot_capacity_tradeoff",
    "plot_latency_vs_units",
    "plot_required_units_distribution",
    "plot_slack_tradeoff",
    "simulate_capacity",
    "vertex_profile",
]
