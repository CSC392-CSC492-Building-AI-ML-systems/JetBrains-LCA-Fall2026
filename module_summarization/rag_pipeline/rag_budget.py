"""Persistent, shared USD cap; reserve a conservative bound before each API call."""
from pathlib import Path
import json
import sqlite3

# Standard synchronous pricing, checked against official model pages 2026-10-22.
GPT41_MINI_SNAPSHOT = 'gpt-4.1-mini-2025-04-14'
GPT41_MINI_CONTEXT_WINDOW = 1_047_576
PRICES = {
    'gpt-3.5-turbo-0125': (.50, 1.50),
    'gpt-4.1-mini': (.40, 1.60),
    GPT41_MINI_SNAPSHOT: (.40, 1.60),
}


def input_token_upper_bound(messages, tools=None):
    # UTF-8 byte bound plus 512 tokens for fixed text-only chat framing.
    if tools is not None:
        wire = json.dumps({'messages': messages, 'tools': tools}, ensure_ascii=False)
        return len(wire.encode('utf-8')) + 512
    return sum(len(message['content'].encode('utf-8')) for message in messages) + 512


class BudgetLedger:
    def __init__(self, path, cap=4.0):
        if not 0 < cap <= 4:
            raise ValueError('This experiment permits a cap above zero and at most US$4')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=30)
        self.db.execute('CREATE TABLE IF NOT EXISTS meta (cap REAL NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS calls (job TEXT PRIMARY KEY, model TEXT, charge REAL, state TEXT, usage TEXT)')
        rows = self.db.execute('SELECT cap FROM meta').fetchall()
        if not rows:
            self.db.execute('INSERT INTO meta VALUES (?)', (cap,))
        elif rows[0][0] != cap:
            raise ValueError('Ledger cap is fixed; do not reset the budget between runs')
        self.cap = cap
        self.db.commit()

    def reserve(self, job, model, messages, max_output, *, tools=None):
        if model not in PRICES:
            raise ValueError('No verified price entry for this model')
        if not isinstance(max_output, int) or not 1 <= max_output <= 2000:
            raise ValueError('Output cap must be an integer from 1 to 2000')
        input_bound = input_token_upper_bound(messages, tools)
        input_rate, output_rate = PRICES[model]
        reserve = (input_bound * input_rate + max_output * output_rate) / 1_000_000
        self.db.execute('BEGIN IMMEDIATE')
        try:
            previous = self.db.execute('SELECT state FROM calls WHERE job=?', (job,)).fetchone()
            if previous:
                raise RuntimeError('Request already recorded; reconcile its output or reservation before retrying')
            used = self.db.execute('SELECT COALESCE(SUM(charge),0) FROM calls').fetchone()[0]
            if used + reserve > self.cap:
                raise RuntimeError('Budget stop: next worst-case request would exceed the shared USD cap')
            self.db.execute('INSERT INTO calls VALUES (?,?,?,?,?)', (job, model, reserve, 'reserved', '{}'))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return reserve

    def settle(self, job, usage):
        row = self.db.execute('SELECT model,charge,state,usage FROM calls WHERE job=?', (job,)).fetchone()
        if row and row[2] == 'complete' and json.loads(row[3]) == usage:
            return row[1]
        if not row or row[2] != 'reserved':
            raise ValueError('No open reservation')
        inp, out = usage.get('prompt_tokens'), usage.get('completion_tokens')
        if not isinstance(inp, int) or not isinstance(out, int) or min(inp, out) < 0:
            raise ValueError('Missing usage: retain the conservative reservation')
        rates = PRICES[row[0]]
        actual = (inp * rates[0] + out * rates[1]) / 1_000_000
        if actual > row[1] + 1e-9:
            raise RuntimeError('Usage exceeded reserve; stop and audit pricing/usage')
        self.db.execute('UPDATE calls SET charge=?,state=?,usage=? WHERE job=?', (actual, 'complete', json.dumps(usage), job))
        self.db.commit()
        return actual

    def reject(self, job):
        # Only use for definite HTTP auth/model/validation rejection, no output.
        self.db.execute("UPDATE calls SET charge=0,state='rejected' WHERE job=? AND state='reserved'", (job,))
        self.db.commit()

    def summary(self):
        rows = self.db.execute('SELECT state,COUNT(*),SUM(charge) FROM calls GROUP BY state').fetchall()
        charges = {state: amount for state, _, amount in rows}
        return {'cap_usd': self.cap, 'confirmed_spend_usd': charges.get('complete', 0),
                'held_for_unresolved_requests_usd': charges.get('reserved', 0),
                'remaining_after_reservations_usd': self.cap - sum(charges.values()),
                'calls_by_state': {state: count for state, count, _ in rows}}
