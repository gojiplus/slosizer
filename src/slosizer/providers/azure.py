"""Azure OpenAI PTU capacity profiles.

This module provides a factory function for creating Azure OpenAI capacity
profiles using the PTU (Provisioned Throughput Unit) model.

Azure PTU throughput is workload-sensitive and must be calibrated per deployment.
See: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/provisioned-throughput
"""

from slosizer.schema import CapacityProfile


def azure_profile(
    model: str,
    *,
    throughput_per_unit: float,
    purchase_increment: int = 1,
    min_units: int = 1,
    input_weight: float = 1.0,
    cached_input_weight: float = 0.0,
    output_weight: float = 4.0,
    thinking_weight: float = 4.0,
    notes: tuple[str, ...] = (),
) -> CapacityProfile:
    """Create an Azure OpenAI PTU capacity profile.

    Azure PTU capacity varies by workload, so profiles must be calibrated
    using the Azure capacity calculator and benchmark data.

    Args:
        model: Model identifier (e.g., "gpt-4.1").
        throughput_per_unit: Tokens per second per PTU.
        purchase_increment: Minimum PTU increment for purchasing.
        min_units: Minimum number of PTUs.
        input_weight: Token weight for input tokens.
        cached_input_weight: Token weight for cached input tokens.
        output_weight: Token weight for output tokens.
        thinking_weight: Token weight for thinking tokens.
        notes: Additional notes about the profile.

    Returns:
        CapacityProfile configured for Azure OpenAI.
    """
    return CapacityProfile(
        provider="azure",
        model=model,
        unit_name="PTU",
        throughput_per_unit=throughput_per_unit,
        purchase_increment=purchase_increment,
        min_units=min_units,
        input_weight=input_weight,
        cached_input_weight=cached_input_weight,
        output_weight=output_weight,
        thinking_weight=thinking_weight,
        source="User-supplied PTU calibration",
        notes=(
            "Seed this profile from the Azure capacity calculator, then refine it with benchmark results and Azure Monitor telemetry.",
            *notes,
        ),
    )
