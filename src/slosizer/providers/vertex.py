"""Google Cloud Vertex AI capacity profiles.

The reviewed model facts live in ``slosizer/data/vertex.toml`` so new models and
provider changes do not require editing planning logic.
"""

from functools import cache

from slosizer.catalog import load_builtin_capacity_profiles
from slosizer.schema import CapacityProfile


@cache
def _vertex_profiles() -> dict[str, CapacityProfile]:
    """Load and cache the packaged Vertex capacity catalog."""
    return load_builtin_capacity_profiles("vertex")


def available_vertex_profiles() -> list[str]:
    """List reviewed built-in Vertex AI model profiles."""
    return sorted(_vertex_profiles())


def vertex_profile(model: str) -> CapacityProfile:
    """Get a reviewed Vertex AI capacity profile by exact model identifier.

    Args:
        model: Exact model identifier supported for Provisioned Throughput.

    Returns:
        Capacity facts for the specified Vertex model.

    Raises:
        KeyError: If the model is absent from the reviewed catalog.
    """
    try:
        return _vertex_profiles()[model]
    except KeyError as exc:
        available = ", ".join(available_vertex_profiles())
        raise KeyError(
            f"Unknown Vertex profile {model!r}. Available profiles: {available}"
        ) from exc
