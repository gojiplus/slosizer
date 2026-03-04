"""Provider-specific capacity profiles.

This module re-exports provider adapters for Vertex AI and Azure OpenAI.
"""

from slosizer.providers.azure import azure_profile
from slosizer.providers.vertex import available_vertex_profiles, vertex_profile

__all__ = ["available_vertex_profiles", "azure_profile", "vertex_profile"]
