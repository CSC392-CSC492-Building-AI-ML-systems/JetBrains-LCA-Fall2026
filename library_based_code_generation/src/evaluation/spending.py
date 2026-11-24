"""Persistent, transactionally reserved experiment spending. No automatic retries."""

import math
import sqlite3
import uuid


class SpendingLedger:
    def __init__(self, path, cap_usd):
        if not math.isfinite(cap_usd) or cap_usd <= 0:
            raise ValueError("A finite positive --spend-limit-usd is required")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.cap = math.floor(cap_usd * 1_000_000)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, reserved INTEGER NOT NULL, actual INTEGER, status TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS identity (singleton INTEGER PRIMARY KEY CHECK(singleton=1), id TEXT NOT NULL)")
        self.db.execute("INSERT OR IGNORE INTO identity VALUES (1,?)", (str(uuid.uuid4()),))
        self.db.commit()
        self.identity = self.db.execute("SELECT id FROM identity WHERE singleton=1").fetchone()[0]

    def reserve(self, request_id, micro_usd):
        if micro_usd < 0:
            raise ValueError("Negative reservation")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            existing = self.db.execute("SELECT status FROM requests WHERE id=?", (request_id,)).fetchone()
            if existing:
                raise RuntimeError("Request already exists in ledger; recover its artifact or reconcile manually. Never retry unknown billing automatically.")
            used = self.db.execute("SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) FROM requests").fetchone()[0]
            if used + micro_usd > self.cap:
                raise RuntimeError("Spending cap reached; completed outputs remain reusable")
            self.db.execute("INSERT INTO requests VALUES (?,?,NULL,'reserved')", (request_id, micro_usd))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def settle(self, request_id, micro_usd):
        if micro_usd < 0:
            raise ValueError("Negative charge")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT reserved,status FROM requests WHERE id=?", (request_id,)).fetchone()
            if row is None or row[1] != "reserved":
                raise RuntimeError("Cannot settle unreserved/already-settled request")
            self.db.execute("UPDATE requests SET actual=?,status='settled' WHERE id=?", (micro_usd, request_id))
            self.db.commit()
            if micro_usd > row[0]:
                raise RuntimeError("Charge exceeded conservative reservation; stop and review prices/token accounting")
        except BaseException:
            self.db.rollback()
            raise

    def summary(self):
        rows = self.db.execute("SELECT status,COUNT(*),SUM(COALESCE(actual,reserved)) FROM requests GROUP BY status").fetchall()
        return {status: {"requests": count, "usd": value / 1_000_000} for status, count, value in rows}

    def close(self):
        self.db.close()
