from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path

from plant_science.api import JsonApplication
from plant_science.jsonio import load_json
from plant_science.service import TrialService


ROOT = Path(__file__).resolve().parents[1]


def _request(body: dict) -> bytes:
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


class AnalysisApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.app = JsonApplication(TrialService(self.connection))
        for user_id, role in (
            ("operator", "operator"),
            ("stat", "statistician"),
            ("stat-2", "statistician"),
            ("approver", "approver"),
        ):
            self.app.handle("POST", "/users", body=_request({
                "user_id": user_id, "display_name": user_id, "role": role,
            }))
        self.app.handle(
            "POST", "/robots", headers={"X-Actor-Id": "operator"},
            body=_request({"robot_id": "robot-a", "model_name": "A 型", "vendor": "厂商"}),
        )
        self.app.handle(
            "POST", "/builds", headers={"X-Actor-Id": "operator"},
            body=_request({
                "build_id": "build-a", "robot_id": "robot-a",
                "version": "1.0", "content_sha256": "b" * 64,
            }),
        )
        self.protocol = load_json(ROOT / "fixtures" / "demo_protocol.json")
        self.app.handle(
            "POST", "/protocols", headers={"X-Actor-Id": "stat"}, body=_request(self.protocol)
        )
        self.app.handle(
            "POST", "/batches", headers={"X-Actor-Id": "operator"},
            body=_request({
                "batch_id": "batch-a", "protocol_id": "demo-delivery-v1",
                "protocol_version": 1, "build_id": "build-a",
            }),
        )
        self.app.handle(
            "POST", "/batches/batch-a/start", headers={"X-Actor-Id": "operator"},
            body=_request({"expected_revision": 1}),
        )
        self.rows = [
            json.loads(line)
            for line in (ROOT / "fixtures" / "demo_observations.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def tearDown(self) -> None:
        self.connection.close()

    def _complete_analysis(self, rows=None) -> dict:
        self.app.handle(
            "POST", "/batches/batch-a/observations",
            headers={"X-Actor-Id": "operator", "Idempotency-Key": "key-1"},
            body=_request({"observations": self.rows if rows is None else rows}),
        )
        batch = self.connection.execute("SELECT revision FROM batches WHERE batch_id='batch-a'").fetchone()
        self.app.handle(
            "POST", "/batches/batch-a/seal", headers={"X-Actor-Id": "stat"},
            body=_request({"expected_revision": batch["revision"]}),
        )
        claimed = self.app.handle("POST", "/jobs/claim", body=_request({"worker_id": "worker", "lease_seconds": 60}))
        job_id = claimed.body["job"]["job_id"]
        completed = self.app.handle(
            "POST", f"/jobs/{job_id}/complete", headers={"X-Actor-Id": "stat"},
            body=_request({"worker_id": "worker"}),
        )
        self.assertEqual(completed.status, 200)
        return completed.body

    def test_analysis_endpoint_returns_insufficient_conditions_and_conclusion(self) -> None:
        only_clear = [row for row in self.rows if row["stratum_key"] == "clear-aisle"]
        self._complete_analysis(only_clear)
        response = self.app.handle(
            "GET", "/batches/batch-a/analysis", headers={"X-Actor-Id": "stat"}
        )
        self.assertEqual(response.status, 200)
        body = response.body
        self.assertEqual(body["conclusion"], "insufficient")
        self.assertEqual(body["insufficient"][0]["condition"], "stratum_missing")
        self.assertEqual(body["insufficient"][0]["stratum"], "cross-traffic")
        self.assertEqual(body["algorithm_version"], "robot-trials-analysis/2")
        self.assertEqual(len(body["input_sha256"]), 64)

    def test_review_then_approve_flow(self) -> None:
        analysis = self._complete_analysis()
        analysis_id = analysis["analysis_id"]
        blocked = self.app.handle(
            "POST", "/decisions", headers={"X-Actor-Id": "approver"},
            body=_request({
                "batch_id": "batch-a", "analysis_id": analysis_id,
                "decision": "approved", "reason": "跳过复核",
            }),
        )
        self.assertEqual(blocked.status, 409)
        self.assertEqual(blocked.body["error"]["code"], "invalid_state")
        self_review = self.app.handle(
            "POST", f"/analyses/{analysis_id}/review", headers={"X-Actor-Id": "stat"},
            body=_request({"verdict": "confirmed", "note": "自我复核"}),
        )
        self.assertEqual(self_review.status, 403)
        reviewed = self.app.handle(
            "POST", f"/analyses/{analysis_id}/review", headers={"X-Actor-Id": "stat-2"},
            body=_request({"verdict": "confirmed", "note": "复核通过"}),
        )
        self.assertEqual(reviewed.status, 201)
        approved = self.app.handle(
            "POST", "/decisions", headers={"X-Actor-Id": "approver"},
            body=_request({
                "batch_id": "batch-a", "analysis_id": analysis_id,
                "decision": "approved", "reason": "满足规则",
            }),
        )
        self.assertEqual(approved.status, 201)

    def test_reopen_then_backfill_keeps_history(self) -> None:
        analysis = self._complete_analysis([row for row in self.rows if row["stratum_key"] == "clear-aisle"])
        analysis_id = analysis["analysis_id"]
        self.app.handle(
            "POST", f"/analyses/{analysis_id}/review", headers={"X-Actor-Id": "stat-2"},
            body=_request({"verdict": "confirmed", "note": "确认样本不足"}),
        )
        decided = self.app.handle(
            "POST", "/decisions", headers={"X-Actor-Id": "approver"},
            body=_request({
                "batch_id": "batch-a", "analysis_id": analysis_id,
                "decision": "needs_more_data", "reason": "缺工况",
            }),
        )
        self.assertEqual(decided.status, 201)
        revision = self.connection.execute("SELECT revision FROM batches WHERE batch_id='batch-a'").fetchone()[0]
        reopened = self.app.handle(
            "POST", "/batches/batch-a/reopen", headers={"X-Actor-Id": "operator"},
            body=_request({"expected_revision": revision, "reason": "补录横向人流"}),
        )
        self.assertEqual(reopened.status, 200)
        report = self.app.handle("GET", "/batches/batch-a/report", headers={"X-Actor-Id": "stat"})
        self.assertEqual([item["decision"] for item in report.body["decisions"]], ["needs_more_data"])


if __name__ == "__main__":
    unittest.main()
