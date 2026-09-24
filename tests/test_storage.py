from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from plant_science.storage import connect, initialize, inspect_schema, transaction


class StorageTests(unittest.TestCase):
    def test_initialize_is_repeatable(self) -> None:
        connection = sqlite3.connect(":memory:", isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            initialize(connection)
            initialize(connection)
            summary = inspect_schema(connection)
        finally:
            connection.close()
        self.assertEqual(summary["missing_tables"], [])
        self.assertEqual(summary["schema_version"], "3")

    def test_initialize_migrates_v2_analyses_snapshot_column(self) -> None:
        connection = sqlite3.connect(":memory:", isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.executescript(
                """
                CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO schema_meta(key, value) VALUES('schema_version', '2');
                CREATE TABLE analyses (
                    analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    batch_revision INTEGER NOT NULL,
                    protocol_sha256 TEXT NOT NULL,
                    input_sha256 TEXT NOT NULL,
                    algorithm_version TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    result_json TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            initialize(connection)
            initialize(connection)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(analyses)")}
            summary = inspect_schema(connection)
        finally:
            connection.close()
        self.assertIn("snapshot_json", columns)
        self.assertEqual(summary["schema_version"], "3")
        self.assertEqual(summary["missing_tables"], [])

    def test_transaction_rolls_back_on_error(self) -> None:
        connection = sqlite3.connect(":memory:", isolation_level=None)
        connection.execute("CREATE TABLE items(value TEXT NOT NULL)")
        with self.assertRaises(RuntimeError):
            with transaction(connection):
                connection.execute("INSERT INTO items(value) VALUES('x')")
                raise RuntimeError("stop")
        count = connection.execute("SELECT count(*) FROM items").fetchone()[0]
        connection.close()
        self.assertEqual(count, 0)

    def test_connect_enables_foreign_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = connect(Path(directory) / "test.sqlite3")
            try:
                self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
