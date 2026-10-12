"""FastAPI app for the Long Code Arena experiment dashboard."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import catalog, runner
from .models import (
    CompareReport,
    ConfigFile,
    DraftRequest,
    FailureReport,
    LaunchRequest,
    LogChunk,
    Meta,
    RunDetail,
    RunSummary,
    TreeNode,
)
from .settings import Settings, load_settings
from .sources import SKIP_DIRS
from .store import ResultStore

logging.basicConfig(level=logging.INFO)

CONFIG_SUFFIXES = {".yaml", ".yml"}
CONFIG_SKIP = SKIP_DIRS | {".github", ".idea", ".mypy_cache", ".pytest_cache", "results", "data", "outputs"}
TASK_FOLDERS = {
    "library_based_code_generation": "library_based_code_generation",
    "project_level_code_completion": "project_level_code_completion",
    "bug_localization": "bug_localization",
    "ci-builds-repair": "ci_builds_repair",
    "commit_message_generation": "commit_message_generation",
    "module_summarization": "module_summarization",
}


def _validate_choice(task: str, approach: str) -> None:
    if task not in catalog.TASK_IDS:
        raise HTTPException(422, f"Unknown task '{task}'")
    if approach not in catalog.APPROACH_IDS:
        raise HTTPException(422, f"Unknown approach '{approach}'")


def _config_files(settings: Settings) -> list[ConfigFile]:
    """Every YAML config in the repo (task configs plus dashboard drafts)."""
    root = settings.repo_root
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in CONFIG_SKIP and not d.startswith("."))
        found += [Path(dirpath) / f for f in sorted(filenames) if Path(f).suffix in CONFIG_SUFFIXES]
    drafts = settings.configs_dir
    if drafts.is_dir() and root not in drafts.parents:
        found += sorted(p for p in drafts.rglob("*") if p.suffix in CONFIG_SUFFIXES)

    out = []
    for f in found:
        rel = f.relative_to(root).as_posix() if root in f.parents else str(f)
        try:
            content = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        out.append(ConfigFile(name=f.name, path=rel, content=content, task=TASK_FOLDERS.get(rel.split("/")[0])))
    return out


def _tree(paths: list[str]) -> list[TreeNode]:
    """Folder tree containing only the given file paths."""
    root: dict = {}
    for path in paths:
        node = root
        for part in path.split("/"):
            node = node.setdefault(part, {})

    def build(children: dict, prefix: str) -> list[TreeNode]:
        items = sorted(children.items(), key=lambda kv: (not kv[1], kv[0].lower()))
        return [
            TreeNode(
                name=name,
                path=f"{prefix}{name}",
                type="dir" if sub else "file",
                children=build(sub, f"{prefix}{name}/") if sub else None,
            )
            for name, sub in items
        ]

    return build(root, "")

def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    store = ResultStore(settings)
    app = FastAPI(title="Long Code Arena Dashboard API", version="0.1.0")
    app.state.settings = settings
    app.state.store = store

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/meta", response_model=Meta)
    def meta() -> Meta:
        return Meta(
            tasks=catalog.TASKS,
            approaches=catalog.APPROACHES,
            failure_categories=catalog.FAILURE_CATEGORIES,
            results_dir=str(settings.results_dir),
            sources=store.sources(),
            runner_configured=bool(settings.run_command),
        )

    @app.get("/api/project-tree", response_model=list[TreeNode])
    def project_tree() -> list[TreeNode]:
        """The repo's folders, pruned to the ones that hold YAML configs."""
        return _tree([c.path for c in _config_files(settings)])

    @app.get("/api/configs", response_model=list[ConfigFile])
    def list_configs() -> list[ConfigFile]:
        return _config_files(settings)

    @app.post("/api/configs/drafts", response_model=ConfigFile, status_code=201)
    def save_draft(req: DraftRequest) -> ConfigFile:
        _validate_choice(req.task, req.approach)
        drafts = settings.configs_dir / "drafts"
        drafts.mkdir(parents=True, exist_ok=True)
        stem = runner.slug(req.filename or req.name or f"{req.approach}-{req.task}")
        path = drafts / f"{stem}.yaml"
        content = runner.config_yaml(req)
        path.write_text(content, encoding="utf-8")
        rel = path.relative_to(settings.repo_root).as_posix() if settings.repo_root in path.parents else str(path)
        return ConfigFile(name=path.name, path=rel, content=content, task=req.task)

    @app.get("/api/runs", response_model=list[RunSummary])
    def list_runs(
        task: str | None = None, approach: str | None = None, status: str | None = None
    ) -> list[RunSummary]:
        return store.list_runs(task=task, approach=approach, status=status)

    @app.post("/api/runs", response_model=RunDetail, status_code=201)
    def launch_run(req: LaunchRequest) -> RunDetail:
        _validate_choice(req.task, req.approach)
        record = runner.create_run(settings, req)
        detail = store.get_run(record.id)
        if detail is None:  # sample data still active would hide it; should not happen
            raise HTTPException(500, "Run was created but could not be read back")
        return detail

    @app.get("/api/runs/{run_id}", response_model=RunDetail)
    def get_run(run_id: str) -> RunDetail:
        detail = store.get_run(run_id)
        if detail is None:
            raise HTTPException(404, f"Run '{run_id}' not found")
        return detail

    @app.get("/api/runs/{run_id}/log", response_model=LogChunk)
    def get_log(run_id: str, offset: int = Query(0, ge=0)) -> LogChunk:
        chunk = store.read_log(run_id, offset)
        if chunk is None:
            raise HTTPException(404, f"Run '{run_id}' not found")
        return chunk

    @app.get("/api/compare", response_model=CompareReport)
    def compare(task: str, model: str | None = None) -> CompareReport:
        if task not in catalog.TASK_IDS:
            raise HTTPException(422, f"Unknown task '{task}'")
        return store.compare(task, model)

    @app.get("/api/failures", response_model=FailureReport)
    def failures(
        task: str | None = None,
        approach: str | None = None,
        category: str | None = None,
        run_id: str | None = None,
    ) -> FailureReport:
        return store.failures(task=task, approach=approach, category=category, run_id=run_id)

    # Serve the built frontend (npm run build) from the same origin when present.
    dist = settings.frontend_dist
    if (dist / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str) -> FileResponse:
            if full_path.startswith("api/"):
                raise HTTPException(404)
            candidate = (dist / full_path).resolve()
            if full_path and candidate.is_file() and dist.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app


app = create_app()
