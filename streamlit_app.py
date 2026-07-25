"""Streamlit app for slosizer capacity planning."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from slosizer import (
    HybridPricingModel,
    HybridTarget,
    LatencyMetric,
    LatencySLO,
    LatencyTarget,
    PaygoPricing,
    PlanOptions,
    ProvisionedPricing,
    RequestSchema,
    ThroughputTarget,
    from_dataframe,
    plan_capacity,
    plan_hybrid_capacity,
)
from slosizer.providers.vertex import available_vertex_profiles, vertex_profile
from slosizer.simulation import bucket_required_units, simulate_capacity

EXAMPLE_CSV = (
    Path(__file__).parent
    / "examples"
    / "input"
    / "synthetic_request_trace_baseline.csv"
)


def load_example_data() -> pd.DataFrame:
    """Load the bundled example CSV."""
    return pd.read_csv(EXAMPLE_CSV)


def infer_schema(df: pd.DataFrame) -> RequestSchema:
    """Infer RequestSchema from DataFrame columns."""
    cols = set(df.columns)

    time_col = "arrival_s" if "arrival_s" in cols else "ts"
    class_col = "class_name" if "class_name" in cols else None
    input_tokens_col = "input_tokens"
    cached_input_tokens_col = (
        "cached_input_tokens" if "cached_input_tokens" in cols else None
    )
    output_tokens_col = "output_tokens"
    thinking_tokens_col = "thinking_tokens" if "thinking_tokens" in cols else None
    max_output_tokens_col = "max_output_tokens" if "max_output_tokens" in cols else None
    latency_col = (
        "observed_latency_s"
        if "observed_latency_s" in cols and df["observed_latency_s"].notna().any()
        else None
    )

    return RequestSchema(
        time_col=time_col,
        class_col=class_col,
        input_tokens_col=input_tokens_col,
        cached_input_tokens_col=cached_input_tokens_col,
        output_tokens_col=output_tokens_col,
        thinking_tokens_col=thinking_tokens_col,
        max_output_tokens_col=max_output_tokens_col,
        latency_col=latency_col,
    )


def main() -> None:
    """Main Streamlit app."""
    st.set_page_config(page_title="Slosizer", page_icon="", layout="wide")
    st.title("Slosizer - LLM Capacity Planner")

    with st.sidebar:
        st.header("Data Source")
        data_source = st.radio(
            "Choose data source",
            ["Use example data", "Upload CSV"],
            label_visibility="collapsed",
        )

        df = None
        if data_source == "Use example data":
            if EXAMPLE_CSV.exists():
                df = load_example_data()
                st.success(f"Loaded {len(df)} requests from example data")
            else:
                st.error("Example CSV not found")
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} requests")

        st.divider()

        st.header("Profile")
        profiles = available_vertex_profiles()
        selected_profile = st.selectbox(
            "Vertex AI Model",
            profiles,
            index=profiles.index("gemini-2.5-flash")
            if "gemini-2.5-flash" in profiles
            else 0,
        )

        st.divider()

        st.header("Planning Target")
        target_type = st.radio(
            "Target type",
            ["Latency SLO", "Throughput", "Hybrid"],
            label_visibility="collapsed",
        )

        if target_type == "Latency SLO":
            threshold_s = st.number_input(
                "Threshold (seconds)",
                min_value=0.1,
                max_value=60.0,
                value=1.5,
                step=0.1,
            )
            percentile = (
                st.number_input(
                    "Percentile",
                    min_value=0.0,
                    max_value=100.0,
                    value=99.0,
                    step=1.0,
                )
                / 100.0
            )
            metric = st.selectbox(
                "Metric",
                ["e2e", "queue_delay"],
                format_func=lambda x: (
                    "End-to-end" if x == "e2e" else "Queue delay only"
                ),
            )
        elif target_type == "Throughput":
            throughput_percentile = (
                st.number_input(
                    "Percentile",
                    min_value=0.0,
                    max_value=100.0,
                    value=99.0,
                    step=1.0,
                    key="throughput_percentile",
                )
                / 100.0
            )
            max_overload = st.slider(
                "Max overload probability",
                min_value=0.001,
                max_value=0.10,
                value=0.01,
                step=0.001,
                format="%.3f",
            )
        else:
            st.subheader("Hybrid Strategy")
            hybrid_strategy = st.radio(
                "Strategy",
                ["cost_optimal", "percentile_split"],
                format_func=lambda x: (
                    "Cost Optimal" if x == "cost_optimal" else "Percentile Split"
                ),
                label_visibility="collapsed",
            )
            if hybrid_strategy == "percentile_split":
                hybrid_percentile = (
                    st.number_input(
                        "Provision Percentile",
                        min_value=1.0,
                        max_value=99.0,
                        value=50.0,
                        step=1.0,
                        key="hybrid_percentile",
                    )
                    / 100.0
                )
            else:
                hybrid_percentile = None

            use_hybrid_slo = st.checkbox("Apply Latency SLO Constraint", value=False)
            if use_hybrid_slo:
                hybrid_slo_threshold = st.number_input(
                    "SLO Threshold (seconds)",
                    min_value=0.1,
                    max_value=60.0,
                    value=1.5,
                    step=0.1,
                    key="hybrid_slo_threshold",
                )
                hybrid_slo_percentile = (
                    st.number_input(
                        "SLO Percentile",
                        min_value=0.0,
                        max_value=100.0,
                        value=99.0,
                        step=1.0,
                        key="hybrid_slo_percentile",
                    )
                    / 100.0
                )
            else:
                hybrid_slo_threshold = None
                hybrid_slo_percentile = None

            st.subheader("Pricing")
            provisioned_cost = st.number_input(
                "Provisioned cost ($/unit/hour)",
                min_value=0.0,
                max_value=100.0,
                value=2.50,
                step=0.01,
                format="%.2f",
            )
            paygo_input_cost = st.number_input(
                "PayGo input ($/M tokens)",
                min_value=0.0,
                max_value=100.0,
                value=0.30,
                step=0.01,
                format="%.2f",
            )
            paygo_output_cost = st.number_input(
                "PayGo output ($/M tokens)",
                min_value=0.0,
                max_value=100.0,
                value=2.50,
                step=0.01,
                format="%.2f",
            )
            priority_multiplier = st.number_input(
                "Priority tier multiplier",
                min_value=1.0,
                max_value=5.0,
                value=1.8,
                step=0.1,
                format="%.1f",
            )

        st.divider()

        run_button = st.button(
            "Plan Capacity", type="primary", use_container_width=True
        )

    if df is not None and run_button:
        try:
            schema = infer_schema(df)
            trace = from_dataframe(df, schema=schema)
            profile = vertex_profile(selected_profile)

            options = PlanOptions()

            if target_type == "Hybrid":
                hybrid_slo = None
                if use_hybrid_slo and hybrid_slo_threshold is not None:
                    hybrid_slo = LatencySLO(
                        threshold_s=hybrid_slo_threshold,
                        percentile=hybrid_slo_percentile,
                    )
                hybrid_target = HybridTarget(
                    strategy=hybrid_strategy,
                    provision_percentile=hybrid_percentile,
                    latency_slo=hybrid_slo,
                )
                pricing = HybridPricingModel(
                    provisioned=ProvisionedPricing(cost_per_unit_hour=provisioned_cost),
                    paygo=PaygoPricing(
                        input_cost_per_million=paygo_input_cost * priority_multiplier,
                        output_cost_per_million=paygo_output_cost * priority_multiplier,
                    ),
                )

                with st.spinner("Planning hybrid capacity..."):
                    hybrid_result = plan_hybrid_capacity(
                        trace, profile, pricing, hybrid_target, options=options
                    )

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Provisioned",
                        f"{hybrid_result.provisioned_units} {hybrid_result.unit_name}s",
                    )
                with col2:
                    st.metric(
                        "Total Cost/hr",
                        f"${hybrid_result.total_cost_hourly:.2f}",
                    )
                with col3:
                    st.metric(
                        "Savings",
                        f"${hybrid_result.savings_vs_full_provision:.2f}/hr",
                        f"{hybrid_result.savings_percent:.1f}%",
                    )
                with col4:
                    st.metric(
                        "Overflow",
                        f"{hybrid_result.overflow_fraction * 100:.1f}%",
                    )

                st.divider()

                tab1, tab2, tab3 = st.tabs(
                    ["Cost Breakdown", "Comparison", "Distribution"]
                )

                with tab1:
                    st.subheader("Cost Breakdown")
                    cost_data = {
                        "Category": ["Provisioned", "PayGo Overflow", "Total"],
                        "Hourly Cost ($)": [
                            hybrid_result.provisioned_cost_hourly,
                            hybrid_result.paygo_cost_hourly,
                            hybrid_result.total_cost_hourly,
                        ],
                    }
                    st.dataframe(pd.DataFrame(cost_data), use_container_width=True)

                    st.subheader("Overflow Tokens")
                    overflow_data = {
                        "Token Type": ["Input", "Output"],
                        "Tokens/hour": [
                            hybrid_result.overflow_input_tokens_hourly,
                            hybrid_result.overflow_output_tokens_hourly,
                        ],
                    }
                    st.dataframe(pd.DataFrame(overflow_data), use_container_width=True)

                    st.subheader("Slack Summary")
                    st.dataframe(
                        hybrid_result.slack_summary.round(4), use_container_width=True
                    )

                with tab2:
                    st.subheader("Hybrid vs Full Provisioning")
                    comparison_data = {
                        "Metric": [
                            "Provisioned Units",
                            "Hourly Cost ($)",
                            "Monthly Cost ($)",
                        ],
                        "Hybrid": [
                            hybrid_result.provisioned_units,
                            hybrid_result.total_cost_hourly,
                            hybrid_result.total_cost_hourly * 24 * 30,
                        ],
                        "Full Provisioned": [
                            hybrid_result.full_provision_units,
                            hybrid_result.full_provision_cost_hourly,
                            hybrid_result.full_provision_cost_hourly * 24 * 30,
                        ],
                        "Savings": [
                            hybrid_result.full_provision_units
                            - hybrid_result.provisioned_units,
                            hybrid_result.savings_vs_full_provision,
                            hybrid_result.savings_vs_full_provision * 24 * 30,
                        ],
                    }
                    st.dataframe(
                        pd.DataFrame(comparison_data), use_container_width=True
                    )

                with tab3:
                    st.subheader("Required Units Distribution")
                    fig, ax = plt.subplots(figsize=(10, 6))

                    windows_s = (1.0, 5.0, 30.0)
                    required = bucket_required_units(
                        trace.frame,
                        profile,
                        units=0,
                        windows_s=windows_s,
                        output_token_source=options.output_token_source,
                    )

                    for w in windows_s:
                        subset = required[required["window_s"] == float(w)]
                        ax.hist(
                            subset["required_units"],
                            bins=30,
                            alpha=0.6,
                            label=f"{w:g}s window",
                        )

                    ax.axvline(
                        hybrid_result.provisioned_units,
                        linestyle="--",
                        color="green",
                        label=f"provisioned ({hybrid_result.provisioned_units})",
                    )
                    ax.axvline(
                        hybrid_result.full_provision_units,
                        linestyle=":",
                        color="red",
                        label=f"full provision ({hybrid_result.full_provision_units})",
                    )
                    ax.set_xlabel(f"Required {profile.unit_name}s")
                    ax.set_ylabel("Bucket count")
                    ax.set_title("Distribution of Required Capacity (Hybrid Planning)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    st.pyplot(fig)
                    plt.close(fig)

            else:
                if target_type == "Latency SLO":
                    slo = LatencySLO(
                        threshold_s=threshold_s,
                        percentile=percentile,
                        metric=LatencyMetric(metric),
                    )
                    target = LatencyTarget(slo=slo)
                else:
                    target = ThroughputTarget(
                        percentile=throughput_percentile,
                        max_overload_probability=max_overload,
                    )

                with st.spinner("Planning capacity..."):
                    result = plan_capacity(trace, profile, target, options=options)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Recommended Capacity",
                        f"{result.recommended_units} {result.unit_name}s",
                    )
                with col2:
                    st.metric("Objective", result.objective.capitalize())
                with col3:
                    st.metric("Target", result.target)

                st.divider()

                tab1, tab2, tab3 = st.tabs(
                    ["Metrics", "Latency vs Units", "Distribution"]
                )

                with tab1:
                    st.subheader("Planning Metrics")
                    metrics_df = pd.DataFrame([result.metrics]).T
                    metrics_df.columns = ["Value"]
                    st.dataframe(metrics_df.round(4), use_container_width=True)

                    if result.latency_summary is not None:
                        st.subheader("Latency Summary")
                        st.dataframe(
                            result.latency_summary.round(4), use_container_width=True
                        )

                    st.subheader("Slack Summary")
                    st.dataframe(
                        result.slack_summary.round(4), use_container_width=True
                    )

                with tab2:
                    st.subheader("Latency vs Provisioned Units")
                    fig, ax = plt.subplots(figsize=(10, 6))

                    min_units = max(1, result.recommended_units - 5)
                    max_units = result.recommended_units + 10
                    units_range = range(min_units, max_units + 1)

                    rows = []
                    for unit in units_range:
                        simulation = simulate_capacity(
                            trace, profile, units=unit, options=options
                        )
                        summary = simulation.latency_summary.iloc[0]
                        rows.append(
                            {
                                "units": unit,
                                "p95_latency_s": float(summary["p95_latency_s"]),
                                "p99_latency_s": float(summary["p99_latency_s"]),
                            }
                        )

                    plot_df = pd.DataFrame(rows)
                    ax.plot(
                        plot_df["units"],
                        plot_df["p95_latency_s"],
                        marker="o",
                        label="p95 latency",
                    )
                    ax.plot(
                        plot_df["units"],
                        plot_df["p99_latency_s"],
                        marker="s",
                        label="p99 latency",
                    )

                    if target_type == "Latency SLO":
                        ax.axhline(
                            threshold_s,
                            linestyle="--",
                            color="red",
                            label="target threshold",
                        )

                    ax.axvline(
                        result.recommended_units,
                        linestyle=":",
                        color="green",
                        label=f"recommended ({result.recommended_units})",
                    )
                    ax.set_xlabel(f"Provisioned {profile.unit_name}s")
                    ax.set_ylabel("Latency (seconds)")
                    ax.set_title("Latency vs Provisioned Capacity")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    st.pyplot(fig)
                    plt.close(fig)

                with tab3:
                    st.subheader("Required Units Distribution")
                    fig, ax = plt.subplots(figsize=(10, 6))

                    windows_s = (1.0, 5.0, 30.0)
                    required = bucket_required_units(
                        trace.frame,
                        profile,
                        units=0,
                        windows_s=windows_s,
                        output_token_source=options.output_token_source,
                    )

                    for window_s in windows_s:
                        subset = required[required["window_s"] == float(window_s)]
                        ax.hist(
                            subset["required_units"],
                            bins=30,
                            alpha=0.6,
                            label=f"{window_s:g}s window",
                        )

                    ax.axvline(
                        result.recommended_units,
                        linestyle="--",
                        color="red",
                        label=f"recommended ({result.recommended_units})",
                    )
                    ax.set_xlabel(f"Required {profile.unit_name}s")
                    ax.set_ylabel("Bucket count")
                    ax.set_title("Distribution of Required Reserved Capacity")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    st.pyplot(fig)
                    plt.close(fig)

        except Exception as e:
            st.error(f"Error: {e}")

    elif df is None:
        st.info("Select a data source and click 'Plan Capacity' to begin.")


if __name__ == "__main__":
    main()
