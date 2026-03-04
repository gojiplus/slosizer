"""Tests for provider adapters."""

import pytest

from slosizer.providers.azure import azure_profile
from slosizer.providers.vertex import available_vertex_profiles, vertex_profile
from slosizer.schema import CapacityProfile


class TestVertexProfiles:
    def test_available_profiles_not_empty(self):
        profiles = available_vertex_profiles()
        assert len(profiles) > 0
        assert isinstance(profiles, list)
        assert all(isinstance(p, str) for p in profiles)

    def test_available_profiles_sorted(self):
        profiles = available_vertex_profiles()
        assert profiles == sorted(profiles)

    def test_get_valid_profile(self):
        profiles = available_vertex_profiles()
        for model in profiles:
            profile = vertex_profile(model)
            assert isinstance(profile, CapacityProfile)
            assert profile.provider == "vertex"
            assert profile.model == model
            assert profile.unit_name == "GSU"
            assert profile.throughput_per_unit is not None
            assert profile.throughput_per_unit > 0

    def test_invalid_profile_raises(self):
        with pytest.raises(KeyError, match="Unknown Vertex profile"):
            vertex_profile("nonexistent-model")

    def test_gemini_25_flash_profile(self):
        profile = vertex_profile("gemini-2.5-flash")

        assert profile.throughput_per_unit == 2690.0
        assert profile.input_weight == 1.0
        assert profile.output_weight == 9.0
        assert profile.thinking_weight == 9.0
        assert profile.long_input_threshold == 200_000
        assert profile.long_input_input_weight == 2.0
        assert profile.long_input_output_weight == 12.0

    def test_gemini_25_pro_profile(self):
        profile = vertex_profile("gemini-2.5-pro")

        assert profile.throughput_per_unit == 650.0
        assert profile.output_weight == 8.0
        assert profile.long_input_threshold == 200_000

    def test_gemini_31_flash_lite_has_cached_weight(self):
        profile = vertex_profile("gemini-3.1-flash-lite-preview")

        assert profile.cached_input_weight == 0.1
        assert profile.output_weight == 6.0

    def test_profiles_have_valid_weights(self):
        for model in available_vertex_profiles():
            profile = vertex_profile(model)
            assert profile.input_weight >= 0
            assert profile.cached_input_weight >= 0
            assert profile.output_weight >= 0
            assert profile.thinking_weight >= 0
            assert profile.purchase_increment >= 1
            assert profile.min_units >= 1


class TestAzureProfile:
    def test_basic_profile(self):
        profile = azure_profile(
            "gpt-4.1",
            throughput_per_unit=12000.0,
        )

        assert profile.provider == "azure"
        assert profile.model == "gpt-4.1"
        assert profile.unit_name == "PTU"
        assert profile.throughput_per_unit == 12000.0

    def test_custom_weights(self):
        profile = azure_profile(
            "gpt-4.1",
            throughput_per_unit=12000.0,
            input_weight=1.0,
            cached_input_weight=0.5,
            output_weight=3.0,
            thinking_weight=3.0,
        )

        assert profile.input_weight == 1.0
        assert profile.cached_input_weight == 0.5
        assert profile.output_weight == 3.0
        assert profile.thinking_weight == 3.0

    def test_custom_purchase_constraints(self):
        profile = azure_profile(
            "gpt-4.1",
            throughput_per_unit=12000.0,
            purchase_increment=50,
            min_units=100,
        )

        assert profile.purchase_increment == 50
        assert profile.min_units == 100

    def test_custom_notes(self):
        profile = azure_profile(
            "gpt-4.1",
            throughput_per_unit=12000.0,
            notes=("Calibrated on 2024-01-01",),
        )

        assert "Calibrated on 2024-01-01" in profile.notes
        # Should also have default note
        assert any("Azure capacity calculator" in note for note in profile.notes)

    def test_source_is_user_supplied(self):
        profile = azure_profile(
            "gpt-4.1",
            throughput_per_unit=12000.0,
        )

        assert "User-supplied" in profile.source
