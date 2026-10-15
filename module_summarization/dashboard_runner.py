"""Run the module-summarization baseline from the experiment dashboard.

The dashboard supplies a generic run configuration through ``LCA_RUN_CONFIG``.
The baseline itself uses the task-specific replication YAML files, so this
adapter selects one of those files and mirrors the baseline result into the
dashboard run record.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "replication_configs" / "config_openai_smoke.yaml"
RUN_FILE = "run.json"


def _run_path() -> Path:
    value = os.environ.get("LCA_RUN_DIR")
    if not value:
        raise RuntimeError("LCA_RUN_DIR must be set when running from the dashboard.")
    return Path(value).expanduser().resolve()


def _read_record(run_dir: Path) -> dict[str, Any]:
    path = run_dir / RUN_FILE
    if not path.is_file():
        raise RuntimeError(f"Dashboard run record was not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_record(run_dir: Path, record: dict[str, Any]) -> None:
    path = run_dir / RUN_FILE
    temporary = run_dir / f".{RUN_FILE}.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _set_stage(record: dict[str, Any], index: int, status: str, detail: str | None = None) -> None:
    stages = record.get("stages", [])
    if index >= len(stages):
        return
    stages[index]["status"] = status
    if detail is not None:
        stages[index]["detail"] = detail


def _config_path() -> Path:
    configured = os.environ.get("LCA_MODULE_SUMMARIZATION_CONFIG")
    if not configured:
        return DEFAULT_CONFIG

    path = Path(configured).expanduser()
    candidates = [path] if path.is_absolute() else [
        Path.cwd() / path,
        MODULE_DIR / path,
        MODULE_DIR.parent / path,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise RuntimeError(f"Module summarization config was not found: {configured}")


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise RuntimeError(f"Module summarization config must contain a YAML mapping: {path}")
    return config


def _apply_dashboard_settings(config: dict[str, Any], run_record: dict[str, Any]) -> dict[str, Any]:
    settings = run_record.get("settings") or {}
    if not isinstance(settings, dict):
        raise RuntimeError("Dashboard run settings must be a mapping.")

    effective = dict(config)
    for key in ("limit", "max_context_toks", "skip_existing"):
        if key in settings:
            effective[key] = settings[key]

    model = run_record.get("model")
    if isinstance(model, str) and model.startswith("gpt-"):
        effective["model_name"] = model
    return effective


def _write_effective_config(run_dir: Path, config: dict[str, Any]) -> Path:
    path = run_dir / "module_summarization_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _update_metrics(run_dir: Path, record: dict[str, Any], summary_path: Path) -> None:
    if not summary_path.is_file():
        raise RuntimeError(f"Baseline completed without writing its summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rouge = summary.get("rouge_l_f1") or {}
    metrics = record.setdefault("metrics", {})
    if isinstance(rouge, dict):
        metrics["rouge_l_f1"] = rouge.get("mean")
    record["instances"] = summary.get("num_predictions")
    record["observation"] = (
        "ROUGE-L F1 is a local diagnostic. Run CompScore separately for the official metric."
    )


def main() -> int:
    run_dir = _run_path()
    record = _read_record(run_dir)
    config_path = _config_path()
    config = _apply_dashboard_settings(_load_config(config_path), record)
    effective_config = _write_effective_config(run_dir, config)

    _set_stage(record, 0, "completed", f"Loaded {config_path.name}")
    _set_stage(record, 1, "completed", "Built relevant-code context")
    _set_stage(record, 2, "running", f"Generating with {config['model_name']}")
    _write_record(run_dir, record)

    command = [sys.executable, str(MODULE_DIR / "baseline_replication.py"), "--config", str(effective_config)]
    process = subprocess.run(command, cwd=MODULE_DIR, check=False)
    if process.returncode != 0:
        record = _read_record(run_dir)
        _set_stage(record, 2, "failed", f"Baseline exited with code {process.returncode}")
        _set_stage(record, 3, "skipped")
        _write_record(run_dir, record)
        return process.returncode

    record = _read_record(run_dir)
    _set_stage(record, 2, "completed")
    _set_stage(record, 3, "completed", "Wrote prediction summary")
    _update_metrics(run_dir, record, MODULE_DIR / config["summary_path"])
    _write_record(run_dir, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
