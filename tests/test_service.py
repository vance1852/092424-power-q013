from __future__ import annotations

import json
import sqlite3
import unittest
from datetime import datetime, timezone
from pathlib import Path

from plant_science.analysis import ALGORITHM_VERSION
from plant_science.clock import FrozenClock
from plant_science.errors import Conflict, Forbidden, InvalidState, ValidationFailed
from plant_science.jsonio import load_json
from plant_science.service import TrialService


ROOT = Path(__file__).resolve().parents[1]


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        self.service = TrialService(self.connection, self.clock)
        for user_id, role in (
            ("operator", "operator"),
            ("stat", "statistician"),
            ("stat-2", "statistician"),
            ("approver", "approver"),
            ("auditor", "auditor"),
        ):
            self.service.create_user(user_id, user_id, role)
        self.protocol = load_json(ROOT / "fixtures" / "demo_protocol.json")
        self.rows = [
            json.loads(line)
            for line in (ROOT / "fixtures" / "demo_observations.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.service.register_robot("operator", "robot-a", "A 型", "厂商")
        self.service.register_build("operator", "build-a", "robot-a", "1.0", "b" * 64)
        self.service.publish_protocol("stat", self.protocol)
        self.service.create_batch("operator", "batch-a", "demo-delivery-v1", 1, "build-a")
        self.service.start_batch("operator", "batch-a", 1)

    def tearDown(self) -> None:
        self.connection.close()

    def test_complete_workflow(self) -> None:
        imported = self.service.import_observations("operator", "batch-a", "key-1", self.rows)
        self.assertEqual(imported["inserted"], 6)
        self.service.seal_batch("stat", "batch-a", 2)
        job = self.service.claim_job("worker", 30)
        analysis = self.service.complete_job("worker", job["job_id"], "stat")
        self.service.review_analysis("stat-2", analysis["analysis_id"], "confirmed", "复核通过")
        self.service.decide("approver", "batch-a", analysis["analysis_id"], "approved", "满足规则")
        report = self.service.report("auditor", "batch-a")
        self.assertEqual(report["batch"]["state"], "decided")
        self.assertEqual(report["analysis"]["result"]["conclusion"], "pass")
        self.assertEqual(len(report["analysis"]["input_snapshot"]), 6)
        self.assertEqual(report["analysis"]["reviews"][0]["reviewed_by"], "stat-2")

    def test_idempotent_replay_and_conflict(self) -> None:
        first = self.service.import_observations("operator", "batch-a", "key-1", self.rows)
        second = self.service.import_observations("operator", "batch-a", "key-1", self.rows)
        self.assertEqual(first, second)
        changed = [dict(item) for item in self.rows]
        changed[0] = dict(changed[0])
        changed[0]["metrics"] = dict(changed[0]["metrics"])
        changed[0]["metrics"]["completion_seconds"] = "99"
        with self.assertRaises(Conflict):
            self.service.import_observations("operator", "batch-a", "key-1", changed)
        count = self.connection.execute("SELECT count(*) FROM observations").fetchone()[0]
        self.assertEqual(count, 6)

    def test_import_rolls_back_when_one_source_row_duplicates(self) -> None:
        self.service.import_observations("operator", "batch-a", "key-1", self.rows[:1])
        with self.assertRaises(Conflict):
            self.service.import_observations("operator", "batch-a", "key-2", self.rows[:2])
        count = self.connection.execute("SELECT count(*) FROM observations").fetchone()[0]
        self.assertEqual(count, 1)

    def test_role_separation(self) -> None:
        with self.assertRaises(Forbidden):
            self.service.seal_batch("operator", "batch-a", 2)
        with self.assertRaises(Forbidden):
            self.service.report("operator", "batch-a")

    def test_exclusion_review_and_revoke_leave_history(self) -> None:
        self.service.import_observations("operator", "batch-a", "key-1", self.rows)
        observation_id = self.connection.execute(
            "SELECT observation_id FROM observations ORDER BY observation_id LIMIT 1"
        ).fetchone()[0]
        requested = self.service.request_exclusion("operator", observation_id, "现场记录失效")
        reviewed = self.service.review_exclusion("stat", requested["exclusion_id"], True, "证据充分")
        self.assertEqual(reviewed["status"], "approved")
        revoked = self.service.revoke_exclusion("operator", requested["exclusion_id"], "已找回原始记录")
        self.assertEqual(revoked["status"], "revoked")
        events = self.connection.execute(
            "SELECT event_type FROM audit_events WHERE entity_type='observation' AND entity_id=? ORDER BY event_id",
            (str(observation_id),),
        ).fetchall()
        self.assertEqual([row[0] for row in events], ["exclusion.requested", "exclusion.revoked"])

    def test_failed_job_returns_to_queue_after_delay(self) -> None:
        self.service.import_observations("operator", "batch-a", "key-1", self.rows)
        self.service.seal_batch("stat", "batch-a", 2)
        job = self.service.claim_job("worker-a", 10)
        failed = self.service.fail_job("worker-a", job["job_id"], "临时计算失败", retry_seconds=5)
        self.assertEqual(failed["state"], "queued")
        self.assertIsNone(self.service.claim_job("worker-b", 10))
        self.clock.advance(seconds=5)
        retried = self.service.claim_job("worker-b", 10)
        self.assertEqual(retried["job_id"], job["job_id"])
        self.assertEqual(retried["attempts"], 2)

    def test_lease_can_be_reclaimed_after_expiry(self) -> None:
        self.service.import_observations("operator", "batch-a", "key-1", self.rows)
        self.service.seal_batch("stat", "batch-a", 2)
        first = self.service.claim_job("worker-a", 10)
        self.clock.advance(seconds=11)
        second = self.service.claim_job("worker-b", 10)
        self.assertEqual(first["job_id"], second["job_id"])
        self.assertEqual(second["lease_owner"], "worker-b")
        with self.assertRaises(InvalidState):
            self.service.complete_job("worker-a", first["job_id"], "stat")

    def _analyzed(self, key: str = "key-1", rows=None) -> dict:
        rows = self.rows if rows is None else rows
        if rows:
            self.service.import_observations("operator", "batch-a", key, rows)
        batch = self.service.get_batch("batch-a")
        self.service.seal_batch("stat", "batch-a", batch["revision"])
        job = self.service.claim_job("worker", 30)
        return self.service.complete_job("worker", job["job_id"], "stat")

    def test_analysis_records_snapshot_and_algorithm_version(self) -> None:
        analysis = self._analyzed()
        self.assertEqual(analysis["algorithm_version"], ALGORITHM_VERSION)
        row = self.connection.execute(
            "SELECT input_snapshot_json,algorithm_version,input_sha256 FROM analyses WHERE analysis_id=?",
            (analysis["analysis_id"],),
        ).fetchone()
        self.assertEqual(row["algorithm_version"], ALGORITHM_VERSION)
        snapshot = json.loads(row["input_snapshot_json"])
        self.assertEqual(len(snapshot), 6)
        self.assertEqual(snapshot[0]["stratum"], "clear-aisle")
        self.assertEqual(row["input_sha256"], analysis["input_sha256"])

    def test_decide_requires_confirmed_review(self) -> None:
        analysis = self._analyzed()
        with self.assertRaises(InvalidState):
            self.service.decide("approver", "batch-a", analysis["analysis_id"], "approved", "跳过复核")

    def test_review_role_separation(self) -> None:
        analysis = self._analyzed()
        with self.assertRaises(Forbidden):
            self.service.review_analysis("stat", analysis["analysis_id"], "confirmed", "自我复核")
        with self.assertRaises(Forbidden):
            self.service.review_analysis("approver", analysis["analysis_id"], "confirmed", "越权复核")
        with self.assertRaises(Forbidden):
            self.service.review_analysis("operator", analysis["analysis_id"], "confirmed", "越权复核")

    def test_review_is_recorded_once_per_reviewer(self) -> None:
        analysis = self._analyzed()
        self.service.review_analysis("stat-2", analysis["analysis_id"], "confirmed", "复核通过")
        with self.assertRaises(Conflict):
            self.service.review_analysis("stat-2", analysis["analysis_id"], "confirmed", "重复复核")

    def test_changes_requested_requires_note(self) -> None:
        analysis = self._analyzed()
        with self.assertRaises(ValidationFailed):
            self.service.review_analysis("stat-2", analysis["analysis_id"], "changes_requested", " ")

    def test_backfill_preserves_historical_decision(self) -> None:
        analysis = self._analyzed()
        self.service.review_analysis("stat-2", analysis["analysis_id"], "confirmed", "复核通过")
        self.service.decide("approver", "batch-a", analysis["analysis_id"], "needs_more_data", "样本量不足")
        with self.assertRaises(InvalidState):
            self.service.import_observations("operator", "batch-a", "key-2", self.rows)
        batch = self.service.get_batch("batch-a")
        reopened = self.service.reopen_batch("operator", "batch-a", batch["revision"], "补录横向人流工况")
        self.assertEqual(reopened["state"], "running")
        extra = [dict(item, source_batch="hall-b-20260924") for item in self.rows]
        self.service.import_observations("operator", "batch-a", "key-2", extra)
        followup = self._analyzed(key="key-3", rows=[])
        self.assertNotEqual(followup["analysis_id"], analysis["analysis_id"])
        self.assertNotEqual(followup["input_sha256"], analysis["input_sha256"])
        self.service.review_analysis("stat-2", followup["analysis_id"], "confirmed", "补录后复核通过")
        self.service.decide("approver", "batch-a", followup["analysis_id"], "approved", "补录后满足规则")
        report = self.service.report("auditor", "batch-a")
        self.assertEqual([item["decision"] for item in report["decisions"]], ["needs_more_data", "approved"])
        self.assertEqual(len(report["analyses"]), 2)
        self.assertEqual(report["decision"]["analysis_id"], followup["analysis_id"])

    def test_reopen_rejected_after_final_decision(self) -> None:
        analysis = self._analyzed()
        self.service.review_analysis("stat-2", analysis["analysis_id"], "confirmed", "复核通过")
        self.service.decide("approver", "batch-a", analysis["analysis_id"], "approved", "满足规则")
        batch = self.service.get_batch("batch-a")
        with self.assertRaises(InvalidState):
            self.service.reopen_batch("operator", "batch-a", batch["revision"], "试图覆盖历史决定")

    def test_stale_analysis_cannot_be_decided_after_reopen(self) -> None:
        analysis = self._analyzed()
        self.service.review_analysis("stat-2", analysis["analysis_id"], "confirmed", "复核通过")
        self.service.decide("approver", "batch-a", analysis["analysis_id"], "needs_more_data", "样本量不足")
        batch = self.service.get_batch("batch-a")
        self.service.reopen_batch("operator", "batch-a", batch["revision"], "补录工况")
        with self.assertRaises(InvalidState):
            self.service.decide("approver", "batch-a", analysis["analysis_id"], "approved", "重复决定")


if __name__ == "__main__":
    unittest.main()
