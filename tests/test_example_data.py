"""Validate the persistent example-data contract."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from slosizer import make_synthetic_trace

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "input"
EXPECTED_SCHEMA = pa.schema(
    [
        pa.field("arrival_s", pa.float64(), nullable=False),
        pa.field("request_id", pa.string(), nullable=False),
        pa.field("class_name", pa.string(), nullable=False),
        pa.field("request_model", pa.string(), nullable=False),
        pa.field("response_model", pa.string(), nullable=False),
        pa.field("service_tier", pa.string(), nullable=False),
        pa.field("input_tokens", pa.int64(), nullable=False),
        pa.field("cached_input_tokens", pa.int64(), nullable=False),
        pa.field("output_tokens", pa.int64(), nullable=False),
        pa.field("thinking_tokens", pa.int64(), nullable=False),
        pa.field("max_output_tokens", pa.int64(), nullable=False),
        pa.field("observed_latency_s", pa.float64(), nullable=True),
        pa.field("business_value", pa.float64(), nullable=True),
    ]
)


def test_examples_use_declared_parquet_schema() -> None:
    paths = sorted(EXAMPLE_DIR.glob("*.parquet"))

    assert [path.name for path in paths] == [
        "synthetic_request_trace_baseline.parquet",
        "synthetic_request_trace_optimized.parquet",
    ]
    assert not list(EXAMPLE_DIR.glob("*.csv"))
    for path in paths:
        assert pq.read_schema(path).remove_metadata() == EXPECTED_SCHEMA
        assert pq.read_metadata(path).num_rows == 23_533


@pytest.mark.parametrize("scenario", ["baseline", "optimized"])
def test_examples_match_seeded_generator(scenario: str) -> None:
    stored = pd.read_parquet(
        EXAMPLE_DIR / f"synthetic_request_trace_{scenario}.parquet"
    )
    generated = make_synthetic_trace(seed=42, scenario=scenario).frame

    pd.testing.assert_frame_equal(stored, generated)
