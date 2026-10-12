"""Readers for the results the task folders of this repo already write.

Each reader turns one task's native output into ``LoadedRun`` objects so the
dashboard can show them next to runs launched from the dashboard itself
(``results/<run_id>/run.json``, see ``store.py``). Repo runs are read-only.

Formats read here:

library_based_code_generation/results/[bm25/]<model path>/
    metadata.json   {"metrics": {"ChrF": {"mean": ..}, "API_recall": {"mean": ..}}, "name": ..}
    <n>.py          one generated program per dataset instance
    (``bm25/`` holds runs that add the top-20 BM25-ranked library names to the prompt.)

ci-builds-repair/**/jobs_results.jsonl  (+ jobs_ids / jobs_invalid / jobs_awaiting)
    one GitHub Actions job per line with its ``conclusion``; written by
    CIFixBenchmark.eval_dataset into its ``out_folder``.

module_summarization/data/**/summary.json  (+ optional CompScore json)
    written by baseline_replication.py and compscore_replication_eval.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .models import Failure, RunRecord, Stage

log = logging.getLogger(__name__)

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "dashboard", "wandb"}


@dataclass
class LoadedRun:
    record: RunRecord
    failures: list[Failure] = field(default_factory=list)
    config_yaml: str | None = None
    log_lines: list[str] = field(default_factory=list)
    # set for dashboard runs, whose files are re-read on demand
    run_dir: Path | None = None


@dataclass(frozen=True)
class Source:
    id: str
    label: str
    path: str  # relative to the repo root
    read: Callable[[Path], list[LoadedRun]]


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9.]+", "-", text).strip("-").lower()


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Skipping unreadable %s: %s", path, exc)
        return None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("Bad JSON line %s:%d", path, n)
    return rows


def _walk(root: Path, filename: str) -> Iterable[Path]:
    """Like ``root.rglob(filename)`` but never descends into SKIP_DIRS."""
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        if filename in filenames:
            yield Path(dirpath) / filename


def _rel(repo: Path, path: Path) -> str:
    return path.relative_to(repo).as_posix()


# ---- library-based code generation -------------------------------------------


def read_library_based(repo: Path) -> list[LoadedRun]:
    root = repo / "library_based_code_generation" / "results"
    runs = []
    for meta_path in _walk(root, "metadata.json"):
        meta = _read_json(meta_path)
        if meta is None:
            continue
        run_dir = meta_path.parent
        parts = run_dir.relative_to(root).parts
        retrieval = parts[0] if parts and parts[0] == "bm25" else None
        model = "/".join(parts[1:] if retrieval else parts)
        metrics_raw = meta.get("metrics", {})

        def mean(key: str) -> float | None:
            value = metrics_raw.get(key)
            value = value.get("mean") if isinstance(value, dict) else value
            return float(value) if isinstance(value, (int, float)) else None

        api_recall, chrf = mean("API_recall"), mean("ChrF")
        predictions = sum(1 for p in run_dir.glob("*.py") if p.stem.isdigit())
        approach = "rag" if retrieval else "baseline"
        settings = (
            {"retrieval": "bm25", "context": "top-20 library method and class names", "query": "instruction"}
            if retrieval
            else {"context": "instruction only"}
        )
        rel = _rel(repo, run_dir)
        runs.append(
            LoadedRun(
                record=RunRecord(
                    id=f"lbcg-{_slug('-'.join(parts))}",
                    name=f"{'BM25 retrieval' if retrieval else 'Instruction only'} · {model.split('/')[-1]}",
                    task="library_based_code_generation",
                    approach=approach,
                    model=model,
                    repo_version="lca-baselines",
                    status="completed",
                    created_at=_mtime_iso(meta_path),
                    settings=settings,
                    metrics={"score": api_recall, "api_recall": api_recall, "chrf": chrf},
                    instances=predictions,
                    observation=(
                        "Prompt adds the top-20 BM25-ranked library names" if retrieval else "Prompt has the instruction only"
                    ),
                    stages=[
                        Stage(name="Generate programs", status="completed", detail=f"{predictions} programs"),
                        Stage(name="Score (ChrF, API Recall)", status="completed"),
                    ],
                    source="library_based_code_generation",
                    source_path=rel,
                ),
                log_lines=[
                    f"> read {rel}/metadata.json",
                    f"> {predictions} generated programs in {rel}/",
                    f"> API Recall {api_recall:.4f}" if api_recall is not None else "> API Recall missing",
                    f"> ChrF {chrf:.4f}" if chrf is not None else "> ChrF missing",
                ],
            )
        )
    return runs


# ---- CI builds repair --------------------------------------------------------

_BRANCH_FN = re.compile(r"__(?P<fn>[a-z0-9_]+?)__id_\d+$")


def read_ci_builds_repair(repo: Path) -> list[LoadedRun]:
    runs = []
    for results_path in _walk(repo / "ci-builds-repair", "jobs_results.jsonl"):
        out = results_path.parent
        results = _read_jsonl(results_path)
        ids = _read_jsonl(out / "jobs_ids.jsonl")
        invalid = _read_jsonl(out / "jobs_invalid.jsonl")
        awaiting = _read_jsonl(out / "jobs_awaiting.jsonl")
        if not results:
            continue

        fns = {m.group("fn") for r in results if (m := _BRANCH_FN.search(r.get("branch_name", "")))}
        fn = fns.pop() if len(fns) == 1 else None
        fix_function = {"diff": "fix_apply_diff", "none": "fix_none"}.get(fn or "", fn)
        model = {"diff": "fix_apply_diff (oracle)", "none": "fix_none (no-op)"}.get(fn or "", fn or "unknown fix function")

        passed = [r for r in results if r.get("conclusion") == "success"]
        metrics: dict[str, float | None] = {
            "score": round(len(passed) / len(results), 4),
            "passed": len(passed),
            "finished_jobs": len(results),
        }
        for d in sorted({r.get("difficulty") for r in results if r.get("difficulty") is not None}):
            group = [r for r in results if r.get("difficulty") == d]
            metrics[f"pass_rate_difficulty_{d}"] = round(sum(r.get("conclusion") == "success" for r in group) / len(group), 4)

        rel = _rel(repo, out)
        run_id = f"ci-{_slug(rel.removeprefix('ci-builds-repair/'))}"
        failures = [
            Failure(
                run_id=run_id,
                task="ci_builds_repair",
                approach="baseline",
                instance_id=f"{r.get('repo_name')}#{r.get('id')}",
                category="uncategorized",
                summary=f"CI on {r.get('repo_name')} concluded '{r.get('conclusion')}' after the fix (difficulty {r.get('difficulty')}).",
                url=r.get("url"),
            )
            for r in results
            if r.get("conclusion") != "success"
        ] + [
            Failure(
                run_id=run_id,
                task="ci_builds_repair",
                approach="baseline",
                instance_id=f"{r.get('repo_name')}#{r.get('id')}",
                category="uncategorized",
                summary=f"Job for {r.get('repo_name')} was marked invalid and never produced a result.",
            )
            for r in invalid
        ]
        runs.append(
            LoadedRun(
                record=RunRecord(
                    id=run_id,
                    name=f"CI build repair · {model}",
                    task="ci_builds_repair",
                    approach="baseline",
                    model=model,
                    repo_version="LCA-CI-fix-benchmark",
                    # the benchmark has finished once results exist; leftovers are reported, not waited on
                    status="completed",
                    created_at=_mtime_iso(results_path),
                    settings={"fix_function": fix_function, "out_folder": rel},
                    metrics=metrics,
                    instances=len(ids) or len(results),
                    observation=f"{len(passed)} of {len(results)} finished builds pass"
                    + (f"; {len(invalid)} invalid" if invalid else "")
                    + (f"; {len(awaiting)} never finished" if awaiting else ""),
                    stages=[
                        Stage(name="Push fixed repos", status="completed", detail=f"{len(ids)} jobs sent"),
                        Stage(name="Run GitHub workflows", status="completed",
                              detail=f"{len(results)} finished" + (f", {len(awaiting)} never finished" if awaiting else "")),
                        Stage(name="Collect conclusions", status="completed",
                              detail=f"{len(invalid)} invalid" if invalid else None),
                    ],
                    source="ci-builds-repair",
                    source_path=rel,
                ),
                failures=failures,
                log_lines=[f"> read {rel}/"]
                + [
                    f"[{r.get('conclusion', '?'):>7}] {r.get('repo_name')} #{r.get('id')} (difficulty {r.get('difficulty')}) {r.get('url', '')}"
                    for r in results
                ],
            )
        )
    return runs


# ---- module summarization ----------------------------------------------------


def read_module_summarization(repo: Path) -> list[LoadedRun]:
    root = repo / "module_summarization"
    compscores = []
    for path in _walk(root, "compscore_results.json"):
        data = _read_json(path)
        if data and data.get("prediction_dir"):
            compscores.append(((root / data["prediction_dir"]).resolve(), data))

    runs = []
    for summary_path in _walk(root / "data", "summary.json"):
        s = _read_json(summary_path)
        if s is None or "num_predictions" not in s:
            continue
        pred_dir = summary_path.parent.resolve()
        comp = next((d for p, d in compscores if p == pred_dir), None)
        rouge = (s.get("rouge_l_f1") or {}).get("mean")
        missing = s.get("missing_prediction_indices") or []
        rel = _rel(repo, summary_path.parent)
        compscore = comp.get("compscore_mean") if comp else None
        runs.append(
            LoadedRun(
                record=RunRecord(
                    id=f"ms-{_slug(rel.removeprefix('module_summarization/data/'))}",
                    name=f"Module summarization · {s.get('model')} · {s.get('max_context_toks')} tokens",
                    task="module_summarization",
                    approach="baseline",
                    model=s.get("model"),
                    repo_version=s.get("dataset"),
                    status="completed",
                    created_at=_mtime_iso(summary_path),
                    settings={"strategy": s.get("strategy"), "max_context_toks": s.get("max_context_toks"),
                              "split": s.get("split")},
                    metrics={"score": compscore, "compscore": compscore, "rouge_l_f1_diagnostic": rouge},
                    instances=s.get("num_predictions"),
                    observation=(
                        f"CompScore over {comp.get('num_examples')} examples" if comp
                        else "CompScore not computed yet; ROUGE-L is a local diagnostic only"
                    ) + (f"; {len(missing)} predictions missing" if missing else ""),
                    stages=[
                        Stage(name="Generate documentation", status="completed",
                              detail=f"{s.get('num_predictions')} / {s.get('num_dataset_rows')}"),
                        Stage(name="Score with CompScore", status="completed" if comp else "waiting"),
                    ],
                    source="module_summarization",
                    source_path=rel,
                ),
                log_lines=[f"> read {rel}/summary.json"] + ([f"> CompScore {compscore:.4f}"] if compscore is not None else []),
            )
        )
    return runs


SOURCES: list[Source] = [
    Source("library_based_code_generation", "Library-based code generation results",
           "library_based_code_generation/results", read_library_based),
    Source("ci-builds-repair", "CI builds repair job results", "ci-builds-repair", read_ci_builds_repair),
    Source("module_summarization", "Module summarization replication outputs",
           "module_summarization/data", read_module_summarization),
]
