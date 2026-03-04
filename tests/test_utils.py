"""Tests for internal utility functions."""

import numpy as np
import pandas as pd
import pytest

from slosizer._utils import adjusted_work, round_up_to_increment, selected_output_tokens
from slosizer.schema import CapacityProfile


class TestRoundUpToIncrement:
    def test_exact_multiple(self):
        assert round_up_to_increment(10.0, 1, 5) == 10

    def test_rounds_up(self):
        assert round_up_to_increment(11.0, 1, 5) == 15

    def test_respects_minimum(self):
        assert round_up_to_increment(2.0, 10, 5) == 10

    def test_minimum_with_increment(self):
        assert round_up_to_increment(2.0, 7, 5) == 10

    def test_zero_units(self):
        assert round_up_to_increment(0.0, 5, 5) == 5


class TestSelectedOutputTokens:
    @pytest.fixture
    def frame(self):
        return pd.DataFrame(
            {
                "output_tokens": [100, 200, 300],
                "max_output_tokens": [500, 500, 500],
            }
        )

    def test_observed_source(self, frame):
        result = selected_output_tokens(frame, "observed")
        np.testing.assert_array_equal(result, [100, 200, 300])

    def test_max_output_source(self, frame):
        result = selected_output_tokens(frame, "max_output_tokens")
        np.testing.assert_array_equal(result, [500, 500, 500])


class TestAdjustedWork:
    @pytest.fixture
    def simple_profile(self):
        return CapacityProfile(
            provider="test",
            model="test-model",
            unit_name="GSU",
            throughput_per_unit=1000.0,
            input_weight=1.0,
            cached_input_weight=0.0,
            output_weight=4.0,
            thinking_weight=4.0,
        )

    @pytest.fixture
    def cached_profile(self):
        return CapacityProfile(
            provider="test",
            model="test-model",
            unit_name="GSU",
            throughput_per_unit=1000.0,
            input_weight=1.0,
            cached_input_weight=0.1,
            output_weight=4.0,
            thinking_weight=4.0,
        )

    @pytest.fixture
    def frame(self):
        return pd.DataFrame(
            {
                "input_tokens": [1000, 2000],
                "cached_input_tokens": [200, 500],
                "output_tokens": [100, 200],
                "thinking_tokens": [50, 100],
                "max_output_tokens": [500, 500],
            }
        )

    def test_basic_calculation(self, simple_profile, frame):
        result = adjusted_work(frame, simple_profile)
        # Row 0: (1000-200)*1.0 + 200*0.0 + 100*4.0 + 50*4.0 = 800 + 0 + 400 + 200 = 1400
        # Row 1: (2000-500)*1.0 + 500*0.0 + 200*4.0 + 100*4.0 = 1500 + 0 + 800 + 400 = 2700
        np.testing.assert_array_almost_equal(result, [1400.0, 2700.0])

    def test_cached_tokens_not_double_counted(self, cached_profile, frame):
        """Verify cached tokens are counted at cached_weight, not input_weight + cached_weight."""
        result = adjusted_work(frame, cached_profile)
        # Row 0: (1000-200)*1.0 + 200*0.1 + 100*4.0 + 50*4.0 = 800 + 20 + 400 + 200 = 1420
        # Row 1: (2000-500)*1.0 + 500*0.1 + 200*4.0 + 100*4.0 = 1500 + 50 + 800 + 400 = 2750
        np.testing.assert_array_almost_equal(result, [1420.0, 2750.0])

    def test_cached_weight_zero_ignores_cached(self, simple_profile, frame):
        """When cached_weight=0, cached tokens still count as input tokens."""
        result = adjusted_work(frame, simple_profile)
        # Non-cached input: (1000-200) = 800 at weight 1.0
        # Cached input: 200 at weight 0.0 = 0
        # Total input contribution: 800 (not 1000)
        # This is correct because cached tokens aren't processed as heavily
        assert result[0] == (1000 - 200) * 1.0 + 200 * 0.0 + 100 * 4.0 + 50 * 4.0

    def test_all_cached(self):
        """When all input is cached, only cached_weight applies."""
        profile = CapacityProfile(
            provider="test",
            model="test-model",
            unit_name="GSU",
            throughput_per_unit=1000.0,
            input_weight=1.0,
            cached_input_weight=0.1,
            output_weight=4.0,
            thinking_weight=4.0,
        )
        frame = pd.DataFrame(
            {
                "input_tokens": [1000],
                "cached_input_tokens": [1000],
                "output_tokens": [100],
                "thinking_tokens": [0],
            }
        )
        result = adjusted_work(frame, profile)
        # (1000-1000)*1.0 + 1000*0.1 + 100*4.0 = 0 + 100 + 400 = 500
        np.testing.assert_array_almost_equal(result, [500.0])

    def test_long_context_threshold(self):
        profile = CapacityProfile(
            provider="test",
            model="test-model",
            unit_name="GSU",
            throughput_per_unit=1000.0,
            input_weight=1.0,
            cached_input_weight=0.0,
            output_weight=4.0,
            thinking_weight=4.0,
            long_input_threshold=5000,
            long_input_input_weight=2.0,
            long_input_output_weight=8.0,
            long_input_thinking_weight=8.0,
        )
        frame = pd.DataFrame(
            {
                "input_tokens": [1000, 10000],
                "cached_input_tokens": [0, 0],
                "output_tokens": [100, 100],
                "thinking_tokens": [50, 50],
            }
        )
        result = adjusted_work(frame, profile)
        # Row 0 (below threshold): 1000*1.0 + 100*4.0 + 50*4.0 = 1600
        # Row 1 (above threshold): 10000*2.0 + 100*8.0 + 50*8.0 = 21200
        np.testing.assert_array_almost_equal(result, [1600.0, 21200.0])

    def test_max_output_tokens_source(self, simple_profile):
        frame = pd.DataFrame(
            {
                "input_tokens": [1000],
                "cached_input_tokens": [0],
                "output_tokens": [100],
                "thinking_tokens": [0],
                "max_output_tokens": [500],
            }
        )
        result_observed = adjusted_work(frame, simple_profile, output_token_source="observed")
        result_max = adjusted_work(frame, simple_profile, output_token_source="max_output_tokens")
        # observed: 1000*1 + 100*4 = 1400
        # max: 1000*1 + 500*4 = 3000
        assert result_observed[0] == 1400.0
        assert result_max[0] == 3000.0
