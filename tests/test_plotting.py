"""Tests for plotting contracts."""

from pathlib import Path

import pandas as pd

from slosizer.plotting import plot_slack_tradeoff


def test_slack_tradeoff_uses_shortest_available_window(tmp_path: Path) -> None:
    comparison = pd.DataFrame(
        {
            "scenario": ["baseline"],
            "target": ["throughput-p99"],
            "avg_spare_fraction_5s": [0.25],
            "avg_spare_fraction_30s": [0.5],
        }
    )
    output = tmp_path / "slack.png"

    plot_slack_tradeoff(comparison, path=output)

    assert output.is_file()
