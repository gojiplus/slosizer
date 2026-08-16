"""Tests for external capacity catalogs."""

from pathlib import Path

import pytest

from slosizer.catalog import load_capacity_profiles


def test_load_capacity_profiles(tmp_path: Path):
    catalog = tmp_path / "profiles.toml"
    catalog.write_text(
        """schema_version = 1
verified_on = 2026-08-15
source = "https://provider.example/capacity"

[[profiles]]
provider = "example"
model = "model-2026-08"
unit_name = "capacity_unit"
throughput_per_unit = 1200.0
purchase_increment = 5
min_units = 10
output_weight = 7.0
""",
        encoding="utf-8",
    )

    profile = load_capacity_profiles(catalog)["model-2026-08"]

    assert profile.provider == "example"
    assert profile.purchase_increment == 5
    assert profile.output_weight == 7.0
    assert profile.verified_on is not None


def test_catalog_schema_version_is_validated(tmp_path: Path):
    catalog = tmp_path / "profiles.toml"
    catalog.write_text("schema_version = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_capacity_profiles(catalog)
