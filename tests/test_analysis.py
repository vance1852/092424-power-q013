from __future__ import annotations

import unittest
from pathlib import Path

from plant_science.analysis import analyze, bootstrap_mean_interval, resample_digest
from plant_science.contracts import Observation
from plant_science.jsonio import load_observations, load_protocol


ROOT = Path(__file__).resolve().parents[1]


def make_row(protocol, source_row, stratum_key, metrics, excluded_reason=None):
    return Observation.from_dict(
        {
            "source_batch": "batch-t",
            "source_row": source_row,
            "robot_id": "robot-a",
            "protocol_id": protocol.protocol_id,
            "protocol_version": protocol.version,
            "stratum_key": stratum_key,
            "observed_at": "2026-09-24T09:00:00+08:00",
            "metrics": metrics,
            "excluded_reason": excluded_reason,
        },
        protocol,
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
        self.assertEqual(result["insufficient"][0]["condition"], "stratum_missing")
        coverage = result["strata"]["cross-traffic"]["coverage"]
        self.assertEqual(coverage["status"], "missing")
        self.assertEqual(coverage["observed"], 0)
        metric = result["strata"]["cross-traffic"]["metrics"]["completion_seconds"]
        self.assertEqual(metric["count"], 0)
        self.assertIsNone(metric["mean"])
        self.assertIsNone(metric["sample_variance"])

    def test_bootstrap_seed_controls_result(self) -> None:
        values = [row.metrics["completion_seconds"] for row in self.rows]
        self.assertEqual(
            bootstrap_mean_interval(values, seed=42, samples=200),
            bootstrap_mean_interval(values, seed=42, samples=200),
        )

    def test_per_stratum_statistics_and_replayable_sampling(self) -> None:
        result = analyze(self.protocol, self.rows)
        metric = result["strata"]["clear-aisle"]["metrics"]["completion_seconds"]
        self.assertEqual(metric["count"], 3)
        self.assertEqual(metric["mean"], "42.33333333333333333333333333")
        self.assertIsNotNone(metric["sample_variance"])
        self.assertLessEqual(metric["bootstrap_lower"], metric["mean"])
        self.assertGreaterEqual(metric["bootstrap_upper"], metric["mean"])
        values = [
            row.metrics["completion_seconds"]
            for row in self.rows
            if row.stratum_key == "clear-aisle"
        ]
        self.assertEqual(
            metric["resample_sha256"],
            resample_digest(values, seed=metric["seed"], samples=metric["bootstrap_samples"]),
        )

    def test_zero_values_are_not_missing(self) -> None:
        rows = tuple(
            make_row(
                self.protocol,
                f"{stratum}-{index}",
                stratum,
                {"completed": 0, "completion_seconds": "50", "interventions": 0},
            )
            for stratum in ("clear-aisle", "cross-traffic")
            for index in range(3)
        )
        result = analyze(self.protocol, rows)
        self.assertEqual(result["insufficient"], [])
        binary = result["strata"]["clear-aisle"]["metrics"]["completed"]
        self.assertEqual(binary["count"], 3)
        self.assertEqual(binary["zero_count"], 3)
        self.assertEqual(binary["missing"], 0)
        self.assertEqual(binary["successes"], 0)
        self.assertTrue(binary["sufficient"])
        self.assertEqual(result["conclusion"], "fail")

    def test_excluded_stratum_is_not_missing_stratum(self) -> None:
        kept = tuple(row for row in self.rows if row.stratum_key == "clear-aisle")
        excluded = tuple(
            make_row(
                self.protocol,
                f"x-{index}",
                "cross-traffic",
                {"completed": 1, "completion_seconds": "60", "interventions": 1},
                excluded_reason="现场记录失效",
            )
            for index in range(2)
        )
        result = analyze(self.protocol, kept + excluded)
        conditions = {item["condition"] for item in result["insufficient"]}
        self.assertIn("stratum_all_excluded", conditions)
        self.assertNotIn("stratum_missing", conditions)
        coverage = result["strata"]["cross-traffic"]["coverage"]
        self.assertEqual(coverage["status"], "all_excluded")
        self.assertEqual(coverage["observed"], 2)
        self.assertEqual(coverage["excluded"], 2)
        self.assertEqual(result["excluded_count"], 2)

    def test_null_metric_value_counts_as_missing_not_zero(self) -> None:
        rows = [
            make_row(
                self.protocol,
                f"c-{index}",
                "clear-aisle",
                {"completed": 1, "completion_seconds": "40", "interventions": 0},
            )
            for index in range(3)
        ]
        rows.append(
            make_row(
                self.protocol,
                "x-0",
                "cross-traffic",
                {"completed": None, "completion_seconds": None, "interventions": None},
            )
        )
        result = analyze(self.protocol, tuple(rows))
        conditions = {item["condition"] for item in result["insufficient"]}
        self.assertIn("coverage_shortfall", conditions)
        self.assertIn("metric_values_missing", conditions)
        metric = result["strata"]["cross-traffic"]["metrics"]["completed"]
        self.assertEqual(metric["count"], 0)
        self.assertEqual(metric["missing"], 1)
        self.assertEqual(metric["zero_count"], 0)
        self.assertEqual(result["missing_value_count"], 3)
        self.assertEqual(result["conclusion"], "insufficient")

    def test_small_sample_is_flagged_per_metric(self) -> None:
        rows = tuple(
            make_row(
                self.protocol,
                f"{stratum}-{index}",
                stratum,
                {"completed": 1, "completion_seconds": "40", "interventions": 0},
            )
            for stratum in ("clear-aisle", "cross-traffic")
            for index in range(2)
        )
        result = analyze(self.protocol, rows)
        shortfalls = [
            item for item in result["insufficient"] if item["condition"] == "metric_coverage_shortfall"
        ]
        self.assertEqual(len(shortfalls), 6)
        self.assertEqual(result["strata"]["clear-aisle"]["metrics"]["completed"]["sufficient"], False)
        self.assertEqual(result["conclusion"], "insufficient")


if __name__ == "__main__":
    unittest.main()
