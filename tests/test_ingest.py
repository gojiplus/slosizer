"""Tests for request trace ingestion."""

import numpy as np
import pandas as pd
import pytest

from slosizer.ingest import from_dataframe
from slosizer.schema import RequestSchema


class TestFromDataframe:
    @pytest.fixture
    def basic_df(self):
        return pd.DataFrame(
            {
                "ts": [0.0, 1.0, 2.0],
                "input_tokens": [100, 200, 300],
                "output_tokens": [50, 100, 150],
            }
        )

    @pytest.fixture
    def full_df(self):
        return pd.DataFrame(
            {
                "ts": [0.0, 1.0, 2.0],
                "class_name": ["chat", "rag", "chat"],
                "input_tokens": [100, 200, 300],
                "cached_input_tokens": [10, 50, 30],
                "output_tokens": [50, 100, 150],
                "thinking_tokens": [0, 20, 0],
                "max_output_tokens": [200, 200, 200],
                "latency_s": [0.5, 1.0, 0.75],
            }
        )

    def test_basic_ingestion(self, basic_df):
        schema = RequestSchema(
            time_col="ts",
            input_tokens_col="input_tokens",
            output_tokens_col="output_tokens",
            class_col=None,
            cached_input_tokens_col=None,
            thinking_tokens_col=None,
            max_output_tokens_col=None,
            latency_col=None,
        )
        trace = from_dataframe(basic_df, schema=schema)

        assert len(trace.frame) == 3
        assert "arrival_s" in trace.frame.columns
        assert "input_tokens" in trace.frame.columns
        assert trace.frame["class_name"].iloc[0] == "default"
        assert trace.frame["cached_input_tokens"].iloc[0] == 0
        assert trace.frame["thinking_tokens"].iloc[0] == 0

    def test_full_ingestion(self, full_df):
        schema = RequestSchema()
        trace = from_dataframe(full_df, schema=schema)

        assert len(trace.frame) == 3
        assert list(trace.frame["class_name"]) == ["chat", "rag", "chat"]
        np.testing.assert_array_equal(trace.frame["cached_input_tokens"], [10, 50, 30])
        np.testing.assert_array_equal(trace.frame["thinking_tokens"], [0, 20, 0])

    def test_arrival_normalization(self):
        df = pd.DataFrame(
            {
                "ts": [100.0, 101.0, 102.0],
                "input_tokens": [100, 200, 300],
                "output_tokens": [50, 100, 150],
            }
        )
        schema = RequestSchema(
            time_col="ts",
            input_tokens_col="input_tokens",
            output_tokens_col="output_tokens",
            class_col=None,
        )
        trace = from_dataframe(df, schema=schema)

        # Should be normalized to start at 0
        assert trace.frame["arrival_s"].iloc[0] == 0.0
        assert trace.frame["arrival_s"].iloc[1] == 1.0
        assert trace.frame["arrival_s"].iloc[2] == 2.0

    def test_datetime_arrival(self):
        df = pd.DataFrame(
            {
                "ts": pd.to_datetime(
                    ["2024-01-01 00:00:00", "2024-01-01 00:00:01", "2024-01-01 00:00:02"]
                ),
                "input_tokens": [100, 200, 300],
                "output_tokens": [50, 100, 150],
            }
        )
        schema = RequestSchema(
            time_col="ts",
            input_tokens_col="input_tokens",
            output_tokens_col="output_tokens",
            class_col=None,
        )
        trace = from_dataframe(df, schema=schema)

        np.testing.assert_array_almost_equal(trace.frame["arrival_s"], [0.0, 1.0, 2.0])

    def test_missing_required_column(self, basic_df):
        schema = RequestSchema(
            time_col="ts",
            input_tokens_col="missing_column",
            output_tokens_col="output_tokens",
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            from_dataframe(basic_df, schema=schema)

    def test_cached_exceeds_input_validation(self):
        df = pd.DataFrame(
            {
                "ts": [0.0],
                "input_tokens": [100],
                "cached_input_tokens": [200],  # exceeds input
                "output_tokens": [50],
            }
        )
        schema = RequestSchema()
        with pytest.raises(ValueError, match="cached_input_tokens cannot exceed"):
            from_dataframe(df, schema=schema)

    def test_negative_tokens_rejected(self):
        df = pd.DataFrame(
            {
                "ts": [0.0],
                "input_tokens": [-100],
                "output_tokens": [50],
            }
        )
        schema = RequestSchema(
            time_col="ts",
            input_tokens_col="input_tokens",
            output_tokens_col="output_tokens",
            class_col=None,
        )
        with pytest.raises(ValueError, match="negative values"):
            from_dataframe(df, schema=schema)

    def test_metadata_preserved(self, basic_df):
        schema = RequestSchema(
            time_col="ts",
            input_tokens_col="input_tokens",
            output_tokens_col="output_tokens",
            class_col=None,
        )
        trace = from_dataframe(
            basic_df,
            schema=schema,
            provider="vertex",
            model="gemini-2.5-flash",
            region="us-central1",
            metadata={"source": "test"},
        )

        assert trace.provider == "vertex"
        assert trace.model == "gemini-2.5-flash"
        assert trace.region == "us-central1"
        assert trace.metadata["source"] == "test"

    def test_sorted_by_arrival(self):
        df = pd.DataFrame(
            {
                "ts": [2.0, 0.0, 1.0],
                "input_tokens": [300, 100, 200],
                "output_tokens": [150, 50, 100],
            }
        )
        schema = RequestSchema(
            time_col="ts",
            input_tokens_col="input_tokens",
            output_tokens_col="output_tokens",
            class_col=None,
        )
        trace = from_dataframe(df, schema=schema)

        # Should be sorted by arrival time
        np.testing.assert_array_equal(trace.frame["input_tokens"], [100, 200, 300])

    def test_default_max_output_tokens(self):
        df = pd.DataFrame(
            {
                "ts": [0.0, 1.0],
                "input_tokens": [100, 200],
                "output_tokens": [50, 150],
            }
        )
        schema = RequestSchema(
            time_col="ts",
            input_tokens_col="input_tokens",
            output_tokens_col="output_tokens",
            class_col=None,
            max_output_tokens_col=None,
        )
        trace = from_dataframe(df, schema=schema)

        # Default should be max observed output tokens
        assert trace.frame["max_output_tokens"].iloc[0] == 150
        assert trace.frame["max_output_tokens"].iloc[1] == 150
