"""Google Cloud Vertex AI capacity profiles.

This module provides built-in capacity profiles for Vertex AI Generative AI
models using the GSU (Generative Service Unit) provisioned throughput model.

See: https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput/supported-models
"""

from slosizer.schema import CapacityProfile

_VERTEX_PROFILES: dict[str, CapacityProfile] = {
    "gemini-2.0-flash-001": CapacityProfile(
        provider="vertex",
        model="gemini-2.0-flash-001",
        unit_name="GSU",
        throughput_per_unit=3360.0,
        purchase_increment=1,
        min_units=1,
        input_weight=1.0,
        cached_input_weight=0.1,
        output_weight=4.0,
        thinking_weight=4.0,
        source="Google Cloud supported models table",
        notes=(
            "Output tokens and separate thinking tokens both use the output-token burndown weight here.",
            "This profile is text-centric. Image, audio, and video tokens need extra columns and weights.",
        ),
    ),
    "gemini-2.0-flash-lite-001": CapacityProfile(
        provider="vertex",
        model="gemini-2.0-flash-lite-001",
        unit_name="GSU",
        throughput_per_unit=6720.0,
        purchase_increment=1,
        min_units=1,
        input_weight=1.0,
        cached_input_weight=0.1,
        output_weight=4.0,
        thinking_weight=4.0,
        source="Google Cloud supported models table",
    ),
    "gemini-2.5-flash": CapacityProfile(
        provider="vertex",
        model="gemini-2.5-flash",
        unit_name="GSU",
        throughput_per_unit=2690.0,
        purchase_increment=1,
        min_units=1,
        input_weight=1.0,
        cached_input_weight=0.1,
        output_weight=9.0,
        thinking_weight=9.0,
        long_input_threshold=200_000,
        long_input_input_weight=2.0,
        long_input_output_weight=12.0,
        long_input_thinking_weight=12.0,
        source="Google Cloud supported models table",
    ),
    "gemini-2.5-flash-lite": CapacityProfile(
        provider="vertex",
        model="gemini-2.5-flash-lite",
        unit_name="GSU",
        throughput_per_unit=8070.0,
        purchase_increment=1,
        min_units=1,
        input_weight=1.0,
        cached_input_weight=0.1,
        output_weight=4.0,
        thinking_weight=4.0,
        source="Google Cloud supported models table",
    ),
    "gemini-2.5-pro": CapacityProfile(
        provider="vertex",
        model="gemini-2.5-pro",
        unit_name="GSU",
        throughput_per_unit=650.0,
        purchase_increment=1,
        min_units=1,
        input_weight=1.0,
        cached_input_weight=0.1,
        output_weight=8.0,
        thinking_weight=8.0,
        long_input_threshold=200_000,
        long_input_input_weight=2.0,
        long_input_output_weight=12.0,
        long_input_thinking_weight=12.0,
        source="Google Cloud supported models table",
    ),
    "gemini-3.1-flash-lite-preview": CapacityProfile(
        provider="vertex",
        model="gemini-3.1-flash-lite-preview",
        unit_name="GSU",
        throughput_per_unit=4030.0,
        purchase_increment=1,
        min_units=1,
        input_weight=1.0,
        cached_input_weight=0.1,
        output_weight=6.0,
        thinking_weight=6.0,
        source="Google Cloud supported models table",
    ),
}


def available_vertex_profiles() -> list[str]:
    """List available built-in Vertex AI model profiles.

    Returns:
        Sorted list of model identifiers.
    """
    return sorted(_VERTEX_PROFILES)


def vertex_profile(model: str) -> CapacityProfile:
    """Get a built-in Vertex AI capacity profile.

    Args:
        model: Model identifier (e.g., "gemini-2.5-flash").

    Returns:
        CapacityProfile configured for the specified Vertex model.

    Raises:
        KeyError: If the model is not in the built-in registry.
    """
    try:
        return _VERTEX_PROFILES[model]
    except KeyError as exc:
        available = ", ".join(available_vertex_profiles())
        raise KeyError(
            f"Unknown Vertex profile {model!r}. Available profiles: {available}"
        ) from exc
