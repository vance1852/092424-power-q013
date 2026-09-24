from __future__ import annotations

import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from plant_science.analysis import analyze, bootstrap_mean_interval
from plant_science.contracts import Observation, Protocol
from plant_science.jsonio import load_json, load_observations, load_protocol


ROOT = Path(__file__).resolve().parents[1]


def make_observation(
    protocol: Protocol,
    stratum: str,
    row: str,
    *,
    completed: str = "1",
    seconds: str = "40.0",
    interventions: str = "0",
    excluded: str | None = None,
) -> Observation:
    return Observation(
        source_batch="batch-t",
        source_row=row,
        robot_id="robot-t",
        protocol_id=protocol.protocol_id,
        protocol_version=protocol.version,
        stratum_key=stratum,
        observed_at="2026-09-21T09:00:00Z",
        metrics={
            "completed": Decimal(completed),
            "completion_seconds": Decimal(seconds),
            "interventions": Decimal(interventions),
        },
        excluded_reason=excluded,
    )


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol(ROOT / "fixtures" / "demo_protocol.json")
        self.rows = load_observations(ROOT / "fixtures" / "demo_observations.jsonl", self.protocol)

    def test_same_snapshot_is_deterministic(self) -> None:
        first = analyze(self.protocol, self.rows)
        second = analyze(self.protocol, self.rows)
        self.assertEqual(first, second)
        self.assertEqual(first["conclusion"], "pass")
        self.assertEqual(first["included_count"], 6)

    def test_missing_stratum_is_insufficient(self) -> None:
        rows = tuple(row for row in self.rows if row.stratum_key == "clear-aisle")
        result = analyze(self.protocol, rows)
        self.assertEqual(result["conclusion"], "insufficient")
        self.assertEqual(result["insufficient"][0]["stratum"], "cross-traffic")

    def test_bootstrap_seed_controls_result(self) -> None:
        values = [row.metrics["completion_seconds"] for row in self.rows]
        self.assertEqual(
            bootstrap_mean_interval(values, seed=42, samples=200),
            bootstrap_mean_interval(values, seed=42, samples=200),
        )

    def test_missing_stratum_is_explicit_and_not_a_zero(self) -> None:
        rows = tuple(row for row in self.rows if row.stratum_key == "clear-aisle")
        result = analyze(self.protocol, rows)
        cross = result["strata"]["cross-traffic"]
        self.assertEqual(cross["status"], "missing")
        self.assertEqual(cross["included"], 0)
        self.assertEqual(cross["excluded"], 0)
        metric = cross["metrics"]["completion_seconds"]
        self.assertEqual(metric["status"], "missing")
        self.assertEqual(metric["count"], 0)
        self.assertEqual(metric["zero_count"], 0)
        self.assertIsNone(metric["mean"])
        self.assertIsNone(metric["sample_variance"])
        self.assertIsNone(metric["ci_lower"])
        conditions = {item["condition"] for item in result["insufficient"]}
        self.assertIn("stratum_missing", conditions)
        self.assertEqual(result["conclusion"], "insufficient")

    def test_zero_values_are_real_data_not_missing(self) -> None:
        rows = tuple(
            make_observation(self.protocol, stratum, f"{stratum}-{index}", completed="0", seconds="0")
            for stratum in ("clear-aisle", "cross-traffic")
            for index in range(3)
        )
        result = analyze(self.protocol, rows)
        self.assertEqual(result["insufficient"], [])
        self.assertEqual(result["conclusion"], "fail")
        for stratum in ("clear-aisle", "cross-traffic"):
            entry = result["strata"][stratum]
            self.assertEqual(entry["status"], "ok")
            metric = entry["metrics"]["completed"]
            self.assertEqual(metric["status"], "ok")
            self.assertEqual(metric["zero_count"], 3)
            self.assertEqual(metric["successes"], 0)
            self.assertEqual(metric["proportion"], 0)

    def test_excluded_observations_are_counted_separately(self) -> None:
        rows = list(self.rows)
        rows[0] = replace(rows[0], excluded_reason="传感器离线")
        result = analyze(self.protocol, tuple(rows))
        clear = result["strata"]["clear-aisle"]
        self.assertEqual(clear["imported"], 3)
        self.assertEqual(clear["excluded"], 1)
        self.assertEqual(clear["included"], 2)
        self.assertEqual(clear["status"], "insufficient")
        self.assertEqual(result["excluded_count"], 1)
        conditions = [item for item in result["insufficient"] if item["condition"] == "coverage_below_required"]
        self.assertEqual(len(conditions), 1)
        self.assertEqual(conditions[0]["stratum"], "clear-aisle")
        self.assertEqual(conditions[0]["actual"], 2)
        self.assertEqual(result["conclusion"], "insufficient")

    def test_single_sample_continuous_metric_is_not_overprecise(self) -> None:
        raw = load_json(ROOT / "fixtures" / "demo_protocol.json")
        raw["strata"] = [dict(item, required_trials=1) for item in raw["strata"]]
        protocol = Protocol.from_dict(raw)
        rows = (
            make_observation(protocol, "clear-aisle", "1"),
            make_observation(protocol, "cross-traffic", "2"),
        )
        result = analyze(protocol, rows)
        self.assertEqual(result["conclusion"], "insufficient")
        metric = result["strata"]["clear-aisle"]["metrics"]["completion_seconds"]
        self.assertEqual(metric["status"], "degenerate")
        self.assertIsNone(metric["sample_variance"])
        self.assertIsNone(metric["ci_lower"])
        conditions = {item["condition"] for item in result["insufficient"]}
        self.assertIn("metric_precision_insufficient", conditions)
        binary = result["strata"]["clear-aisle"]["metrics"]["completed"]
        self.assertEqual(binary["status"], "ok")
        self.assertGreater(binary["ci_upper"], binary["ci_lower"])

    def test_metric_seeds_make_sampling_replayable(self) -> None:
        result = analyze(self.protocol, self.rows)
        metric = result["strata"]["clear-aisle"]["metrics"]["completion_seconds"]
        values = [
            row.metrics["completion_seconds"]
            for row in self.rows
            if row.stratum_key == "clear-aisle"
        ]
        lower, upper = bootstrap_mean_interval(
            values, seed=metric["seed"], samples=self.protocol.bootstrap_samples
        )
        self.assertEqual(metric["ci_lower"], format(lower, "f"))
        self.assertEqual(metric["ci_upper"], format(upper, "f"))
        self.assertEqual(metric["seed"], self.protocol.seed + 1)


if __name__ == "__main__":
    unittest.main()
