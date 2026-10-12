"""Launch evaluation runs.

The dashboard does not run Long Code Arena itself. A launch creates the run
folder (``run.json`` with status ``queued`` plus the exact ``config.yaml``) and,
when ``LCA_RUN_COMMAND`` is set, starts that command in the background with:

    LCA_RUN_DIR      the run folder to write results into
    LCA_RUN_CONFIG   path to its config.yaml
    LCA_RUN_ID       the run id

stdout/stderr are appended to ``log.txt``. The command may update ``run.json``
itself (stages, metrics, evidence) as it goes; when it exits without having set
a final status, the run is marked completed or failed from its exit code.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import catalog
from .models import LaunchRequest, RunRecord, Stage
from .store import CONFIG_FILE, LOG_FILE, RUN_FILE
from .settings import Settings

_lock = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "run"


def config_document(req: LaunchRequest) -> dict[str, Any]:
    return {
        "task": req.task,
        "approach": req.approach,
        "model": req.model,
        "repo_version": req.repo_version,
        **({"name": req.name} if req.name else {}),
        "settings": req.settings,
    }


def config_yaml(req: LaunchRequest) -> str:
    return yaml.safe_dump(config_document(req), sort_keys=False)


def write_run_json(run_dir: Path, record: RunRecord) -> None:
    tmp = run_dir / f".{RUN_FILE}.tmp"
    tmp.write_text(json.dumps(record.model_dump(exclude_none=True), indent=2), encoding="utf-8")
    tmp.replace(run_dir / RUN_FILE)


def _append_log(run_dir: Path, line: str) -> None:
    with open(run_dir / LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line.rstrip("\n") + "\n")


def create_run(settings: Settings, req: LaunchRequest) -> RunRecord:
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = f"{stamp}-{slug(req.approach)}-{slug(req.task)}"
    with _lock:
        run_id, n = base, 1
        while (settings.results_dir / run_id).exists():
            n += 1
            run_id = f"{base}-{n}"
        run_dir = settings.results_dir / run_id
        run_dir.mkdir()

    record = RunRecord(
        id=run_id,
        name=req.name,
        task=req.task,
        approach=req.approach,
        model=req.model,
        repo_version=req.repo_version,
        status="queued",
        created_at=now_iso(),
        settings=req.settings,
        stages=[Stage(name=s) for s in catalog.STAGES_BY_APPROACH.get(req.approach, catalog.DEFAULT_STAGES)],
    )
    (run_dir / CONFIG_FILE).write_text(config_yaml(req), encoding="utf-8")
    write_run_json(run_dir, record)

    if settings.run_command:
        _append_log(run_dir, f"> queued: {' '.join(settings.run_command)}")
        threading.Thread(target=_execute, args=(settings, run_dir), daemon=True).start()
    else:
        _append_log(run_dir, "> queued; no runner configured (set LCA_RUN_COMMAND to execute runs)")
    return record


def _load(run_dir: Path) -> RunRecord:
    return RunRecord.model_validate(json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8")))


def _execute(settings: Settings, run_dir: Path) -> None:
    record = _load(run_dir)
    record.status = "running"
    record.started_at = now_iso()
    write_run_json(run_dir, record)

    env = {
        **os.environ,
        "LCA_RUN_DIR": str(run_dir),
        "LCA_RUN_CONFIG": str(run_dir / CONFIG_FILE),
        "LCA_RUN_ID": run_dir.name,
    }
    try:
        with open(run_dir / LOG_FILE, "a") as log_fh:
            proc = subprocess.Popen(
                settings.run_command,
                cwd=settings.repo_root,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            code = proc.wait()
    except OSError as exc:
        _append_log(run_dir, f"> failed to start runner: {exc}")
        code = -1

    record = _load(run_dir)  # the command may have rewritten it
    if record.status not in ("completed", "failed"):
        record.status = "completed" if code == 0 else "failed"
    record.finished_at = record.finished_at or now_iso()
    if record.runtime_s is None and record.started_at:
        started = datetime.fromisoformat(record.started_at)
        record.runtime_s = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
    if code != 0:
        for stage in record.stages:
            if stage.status == "running":
                stage.status = "failed"
    write_run_json(run_dir, record)
    _append_log(run_dir, f"> runner exited with code {code}")
