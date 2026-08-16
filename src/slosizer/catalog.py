"""Load versioned provider capacity facts from TOML catalogs."""

import tomllib
from collections.abc import Mapping
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Any

from slosizer.schema import CapacityProfile


def _parse_catalog(data: Mapping[str, Any]) -> dict[str, CapacityProfile]:
    """Validate parsed catalog data and construct immutable profiles."""
    if data.get("schema_version") != 1:
        raise ValueError("capacity catalog schema_version must be 1")
    source = str(data.get("source", ""))
    verified_on = data.get("verified_on")
    if verified_on is not None and not isinstance(verified_on, date):
        raise ValueError("capacity catalog verified_on must be a TOML local date")

    raw_profiles = data.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError(
            "capacity catalog must contain at least one [[profiles]] entry"
        )

    profiles: dict[str, CapacityProfile] = {}
    for raw_profile in raw_profiles:
        if not isinstance(raw_profile, dict):
            raise ValueError("each capacity catalog profile must be a table")
        values = dict(raw_profile)
        values.setdefault("source", source)
        values.setdefault("verified_on", verified_on)
        values["notes"] = tuple(values.get("notes", ()))
        profile = CapacityProfile(**values)
        if profile.model in profiles:
            raise ValueError(f"duplicate capacity profile for {profile.model!r}")
        profiles[profile.model] = profile
    return profiles


def load_capacity_profiles(path: str | Path) -> dict[str, CapacityProfile]:
    """Load capacity profiles from a version-controlled TOML file.

    Args:
        path: Path to a catalog with ``schema_version = 1`` and one or more
            ``[[profiles]]`` tables.

    Returns:
        Profiles keyed by exact provider model identifier.
    """
    with Path(path).open("rb") as catalog_file:
        return _parse_catalog(tomllib.load(catalog_file))


def load_builtin_capacity_profiles(name: str) -> dict[str, CapacityProfile]:
    """Load one of the package's reviewed provider catalogs."""
    resource = files("slosizer.data").joinpath(f"{name}.toml")
    return _parse_catalog(tomllib.loads(resource.read_text(encoding="utf-8")))
