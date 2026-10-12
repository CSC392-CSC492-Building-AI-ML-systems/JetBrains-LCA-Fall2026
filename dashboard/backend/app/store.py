"""Collect runs from the repo's own result folders and from dashboard runs.

Two kinds of runs are served:

* Repo runs, read from the files each task already writes (``sources.py``).
* Dashboard runs, one folder per run::

    results/
      <run_id>/
        run.json          # RunRecord (required)
        config.yaml       # the exact config the run used (optional)
        log.txt           # console output, one line per event (optional)
        failures.jsonl    # one failure record per line (optional)

Everything is re-read on every request; the files are small and this keeps the
dashboard live while experiments write to them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from . import catalog
from .models import (
    CompareReport,
    CompareRow,
    Failure,
    FailureReport,
    LeaderboardCell,
    LeaderboardRow,
    LogChunk,
    RunDetail,
    RunRecord,
    RunSummary,
    SourceInfo,
)
from .settings import Settings
from .sources import SOURCES, LoadedRun

log = logging.getLogger(__name__)

RUN_FILE = "run.json"
CONFIG_FILE = "config.yaml"
LOG_FILE = "log.txt"
FAILURES_FILE = "failures.jsonl"


class ResultStore:
    def __init__(self, settings: Settings):
        self.settings = settings

    # ---- dashboard runs -------------------------------------------------

    def _load_dashboard_run(self, run_dir: Path) -> LoadedRun | None:
        try:
            data = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
            data.setdefault("id", run_dir.name)
            data["source"] = "dashboard"
            data["source_path"] = run_dir.relative_to(self.settings.repo_root).as_posix() \
                if self.settings.repo_root in run_dir.parents else str(run_dir)
            record = RunRecord.model_validate(data)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            log.warning("Skipping unreadable run %s: %s", run_dir, exc)
            return None
        return LoadedRun(record=record, failures=self._load_failures(run_dir, record), run_dir=run_dir)

    def _load_failures(self, run_dir: Path, record: RunRecord) -> list[Failure]:
        path = run_dir / FAILURES_FILE
        if not path.is_file():
            return []
        failures: list[Failure] = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                raw.setdefault("run_id", record.id)
                raw.setdefault("task", record.task)
                raw.setdefault("approach", record.approach)
                raw.setdefault("instance_id", f"{record.id}#{n}")
                failures.append(Failure.model_validate(raw))
            except (json.JSONDecodeError, ValidationError) as exc:
                log.warning("Bad failure record %s:%d: %s", path, n, exc)
        return failures

    def _dashboard_runs(self) -> list[LoadedRun]:
        root = self.settings.results_dir
        if not root.is_dir():
            return []
        loaded = (self._load_dashboard_run(p.parent) for p in sorted(root.glob(f"*/{RUN_FILE}")))
        return [r for r in loaded if r is not None]

    # ---- all runs -------------------------------------------------------

    def _repo_runs(self) -> dict[str, list[LoadedRun]]:
        if not self.settings.read_repo_results:
            return {}
        out = {}
        for source in SOURCES:
            try:
                out[source.id] = source.read(self.settings.repo_root)
            except Exception:  # one broken source must not take the dashboard down
                log.exception("Reading %s failed", source.id)
                out[source.id] = []
        return out

    def runs(self) -> list[LoadedRun]:
        """All runs, newest first."""
        runs = self._dashboard_runs() + [r for group in self._repo_runs().values() for r in group]
        runs.sort(key=lambda r: r.record.created_at or "", reverse=True)
        return runs

    def sources(self) -> list[SourceInfo]:
        repo = self._repo_runs()
        infos = [
            SourceInfo(
                id="dashboard",
                label="Runs launched from the dashboard",
                path=str(self.settings.results_dir),
                runs=len(self._dashboard_runs()),
            )
        ]
        infos += [SourceInfo(id=s.id, label=s.label, path=s.path, runs=len(repo.get(s.id, []))) for s in SOURCES]
        return infos

    def find(self, run_id: str) -> LoadedRun | None:
        if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
            return None
        d = self.settings.results_dir / run_id
        if (d / RUN_FILE).is_file():
            return self._load_dashboard_run(d)
        return next((r for r in self.runs() if r.record.id == run_id), None)

    # ---- views ----------------------------------------------------------

    @staticmethod
    def _summary(run: LoadedRun) -> RunSummary:
        rec = run.record
        return RunSummary(
            id=rec.id,
            name=rec.name or f"{catalog.approach_label(rec.approach)} evaluation - {catalog.task_label(rec.task)}",
            task=rec.task,
            task_label=catalog.task_label(rec.task),
            approach=rec.approach,
            approach_label=catalog.approach_label(rec.approach),
            model=rec.model,
            status=rec.status,
            created_at=rec.created_at,
            score=rec.score,
            runtime_s=rec.runtime_s,
            cost_usd=rec.cost_usd,
            failure_count=len(run.failures),
            source=rec.source,
        )

    def list_runs(
        self, task: str | None = None, approach: str | None = None, status: str | None = None
    ) -> list[RunSummary]:
        return [
            self._summary(r)
            for r in self.runs()
            if (task is None or r.record.task == task)
            and (approach is None or r.record.approach == approach)
            and (status is None or r.record.status == status)
        ]

    def get_run(self, run_id: str) -> RunDetail | None:
        run = self.find(run_id)
        if run is None:
            return None
        rec = run.record
        config_yaml = run.config_yaml
        if run.run_dir is not None and (run.run_dir / CONFIG_FILE).is_file():
            config_yaml = (run.run_dir / CONFIG_FILE).read_text(encoding="utf-8")
        return RunDetail(
            **self._summary(run).model_dump(),
            repo_version=rec.repo_version,
            started_at=rec.started_at,
            finished_at=rec.finished_at,
            settings=rec.settings,
            metrics=rec.metrics,
            tokens=rec.tokens,
            instances=rec.instances,
            observation=rec.observation,
            stages=rec.stages,
            evidence=rec.evidence,
            config_yaml=config_yaml,
            source_path=rec.source_path,
        )

    def read_log(self, run_id: str, offset: int = 0) -> LogChunk | None:
        run = self.find(run_id)
        if run is None:
            return None
        if run.run_dir is not None:
            path = run.run_dir / LOG_FILE
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.is_file() else []
        else:
            lines = run.log_lines
        offset = max(0, min(offset, len(lines)))
        return LogChunk(lines=lines[offset:], next_offset=len(lines))

    # ---- aggregates -----------------------------------------------------

    def failures(
        self,
        task: str | None = None,
        approach: str | None = None,
        category: str | None = None,
        run_id: str | None = None,
    ) -> FailureReport:
        items: list[Failure] = []
        for run in self.runs():
            if (task and run.record.task != task) or (run_id and run.record.id != run_id):
                continue
            items.extend(run.failures)

        scoped = [f for f in items if approach is None or f.approach == approach]
        counts = {c.id: 0 for c in catalog.FAILURE_CATEGORIES}
        for f in scoped:
            counts[f.category] = counts.get(f.category, 0) + 1

        by_approach: dict[str, dict[str, int]] = {}
        for f in items:
            bucket = by_approach.setdefault(f.approach, {})
            bucket[f.category] = bucket.get(f.category, 0) + 1

        return FailureReport(
            categories=catalog.FAILURE_CATEGORIES,
            counts=counts,
            counts_by_approach=by_approach,
            items=[f for f in scoped if category is None or f.category == category],
        )

    def compare(self, task: str, model: str | None = None) -> CompareReport:
        """Compare approaches on ``task`` for one model.

        ``rows`` holds the newest completed run of each approach for ``model``
        (deltas are against that model's baseline, so they are like-for-like).
        Without ``model``, the model covering the most approaches is used.
        ``leaderboard`` holds every model x approach for the task.
        """
        info = next((t for t in catalog.TASKS if t.id == task), None)
        order = [a.id for a in catalog.APPROACHES]

        # newest completed run per (model, approach)
        latest: dict[tuple[str, str], LoadedRun] = {}
        for run in self.runs():  # newest first
            rec = run.record
            key = (rec.model or "unknown", rec.approach)
            if rec.task == task and rec.status == "completed" and key not in latest:
                latest[key] = run

        by_model: dict[str, dict[str, LoadedRun]] = {}
        for (m, a), run in latest.items():
            by_model.setdefault(m, {})[a] = run

        def best(m: str) -> float:
            return max((r.record.score or 0.0) for r in by_model[m].values())

        models = sorted(by_model, key=lambda m: (-len(by_model[m]), -best(m), m))
        chosen = model if model in by_model else (models[0] if models else None)

        picked = by_model.get(chosen, {}) if chosen else {}
        approaches = sorted(picked, key=lambda a: order.index(a) if a in order else len(order))
        baseline = picked.get("baseline")
        baseline_score = baseline.record.score if baseline else None

        rows: list[CompareRow] = []
        failures: list[Failure] = []
        for a in approaches:
            run = picked[a]
            rec = run.record
            failures.extend(run.failures)
            rows.append(
                CompareRow(
                    approach=a,
                    approach_label=catalog.approach_label(a),
                    run_id=rec.id,
                    model=rec.model,
                    score=rec.score,
                    delta_vs_baseline=(
                        round(rec.score - baseline_score, 4)
                        if rec.score is not None and baseline_score is not None and a != "baseline"
                        else None
                    ),
                    runtime_s=rec.runtime_s,
                    cost_usd=rec.cost_usd,
                    observation=rec.observation,
                    failure_count=len(run.failures),
                )
            )

        leaderboard = [
            LeaderboardRow(
                model=m,
                cells={a: LeaderboardCell(run_id=r.record.id, score=r.record.score) for a, r in by_model[m].items()},
            )
            for m in sorted(by_model, key=lambda m: -best(m))
        ]

        # Prefer a "context was there but the answer was wrong" example: it is the
        # most informative distinction the error analysis makes.
        example = next(
            (f for f in failures if f.category == "reasoning" and f.retrieved),
            next((f for f in failures if f.category == "reasoning"), failures[0] if failures else None),
        )
        return CompareReport(
            task=task,
            task_label=catalog.task_label(task),
            metric=info.metric if info else "score",
            models=models,
            model=chosen,
            rows=rows,
            leaderboard=leaderboard,
            example_finding=example,
        )
