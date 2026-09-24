from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path

from plant_science.api import JsonApplication
from plant_science.jsonio import load_json
from plant_science.service import TrialService


ROOT = Path(__file__).resolve().parents[1]


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.app = JsonApplication(TrialService(self.connection))

    def tearDown(self) -> None:
        self.connection.close()

    def test_health(self) -> None:
        response = self.app.handle("GET", "/health")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.body["status"], "ok")

    def test_json_error_shape(self) -> None:
        response = self.app.handle("POST", "/users", body=b"not-json")
        self.assertEqual(response.status, 422)
        self.assertEqual(response.body["error"]["code"], "validation_failed")

    def test_user_route(self) -> None:
        payload = json.dumps({"user_id": "u1", "display_name": "操作员", "role": "operator"}).encode()
        response = self.app.handle("POST", "/users", body=payload)
        self.assertEqual(response.status, 201)
        self.assertEqual(response.body["role"], "operator")

    def _prepare_sealed_batch(self, observation_rows: list[dict]) -> None:
        service = self.app.service
        service.create_user("op", "操作员", "operator")
        service.create_user("stat", "统计负责人", "statistician")
        service.create_user("appr", "批准人", "approver")
        service.create_user("aud", "审计", "auditor")
        service.register_robot("op", "robot-a", "A 型", "厂商")
        service.register_build("op", "build-a", "robot-a", "1.0", "c" * 64)
        protocol = load_json(ROOT / "fixtures" / "demo_protocol.json")
        service.publish_protocol("stat", protocol)
        service.create_batch("op", "batch-a", "demo-delivery-v1", 1, "build-a")
        service.start_batch("op", "batch-a", 1)
        service.import_observations("op", "batch-a", "key-1", observation_rows)
        service.seal_batch("stat", "batch-a", 2)

    @staticmethod
    def _observation_rows() -> list[dict]:
        return [
            json.loads(line)
            for line in (ROOT / "fixtures" / "demo_observations.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _complete_via_api(self) -> dict:
        claimed = self.app.handle("POST", "/jobs/claim", body=json.dumps({"worker_id": "w1"}).encode())
        job_id = claimed.body["job"]["job_id"]
        completed = self.app.handle(
            "POST",
            f"/jobs/{job_id}/complete",
            headers={"X-Actor-Id": "stat"},
            body=json.dumps({"worker_id": "w1"}).encode(),
        )
        self.assertEqual(completed.status, 200)
        return completed.body

    def test_complete_returns_insufficient_conditions_and_conclusion(self) -> None:
        rows = [row for row in self._observation_rows() if row["stratum_key"] == "clear-aisle"]
        self._prepare_sealed_batch(rows)
        body = self._complete_via_api()
        self.assertEqual(body["conclusion"], "insufficient")
        conditions = {(item["stratum"], item["condition"]) for item in body["insufficient"]}
        self.assertIn(("cross-traffic", "stratum_missing"), conditions)
        self.assertEqual(body["result"]["conclusion"], "insufficient")

    def test_replay_route_reproduces_analysis(self) -> None:
        self._prepare_sealed_batch(self._observation_rows())
        body = self._complete_via_api()
        self.assertEqual(body["conclusion"], "pass")
        replay = self.app.handle(
            "POST", f"/analyses/{body['analysis_id']}/replay", headers={"X-Actor-Id": "aud"}
        )
        self.assertEqual(replay.status, 200)
        self.assertTrue(replay.body["replay_matches"])
        self.assertTrue(replay.body["snapshot_integrity"])
        denied = self.app.handle(
            "POST", f"/analyses/{body['analysis_id']}/replay", headers={"X-Actor-Id": "op"}
        )
        self.assertEqual(denied.status, 403)

    def test_decision_route_enforces_role_separation(self) -> None:
        self._prepare_sealed_batch(self._observation_rows())
        body = self._complete_via_api()
        payload = {
            "batch_id": "batch-a",
            "analysis_id": body["analysis_id"],
            "decision": "approved",
            "reason": "满足规则",
        }
        denied = self.app.handle(
            "POST", "/decisions", headers={"X-Actor-Id": "stat"}, body=json.dumps(payload).encode()
        )
        self.assertEqual(denied.status, 403)
        approved = self.app.handle(
            "POST", "/decisions", headers={"X-Actor-Id": "appr"}, body=json.dumps(payload).encode()
        )
        self.assertEqual(approved.status, 201)
        report = self.app.handle("GET", "/batches/batch-a/report", headers={"X-Actor-Id": "aud"})
        self.assertEqual(report.body["decision"]["decision"], "approved")
        self.assertEqual(len(report.body["analysis"]["snapshot"]), 6)


if __name__ == "__main__":
    unittest.main()
