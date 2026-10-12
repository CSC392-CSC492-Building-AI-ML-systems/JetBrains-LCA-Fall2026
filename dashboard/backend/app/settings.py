"""Paths and runtime settings, all overridable with environment variables.

By default the dashboard assumes it lives in ``<repo>/dashboard/backend``. It reads
the results the task folders of this repo already produce (see ``sources.py``)
plus dashboard-launched runs in ``<repo>/results``.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ROOT = BACKEND_DIR.parent.parent


def _path(env: str, default: Path) -> Path:
    value = os.environ.get(env)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    results_dir: Path
    configs_dir: Path
    frontend_dist: Path
    run_command: list[str] = field(default_factory=list)
    read_repo_results: bool = True
    cors_origins: list[str] = field(default_factory=list)


def load_settings() -> Settings:
    repo_root = _path("LCA_REPO_ROOT", DEFAULT_REPO_ROOT)
    return Settings(
        repo_root=repo_root,
        results_dir=_path("LCA_RESULTS_DIR", repo_root / "results"),
        configs_dir=_path("LCA_CONFIGS_DIR", repo_root / "configs"),
        frontend_dist=_path("LCA_FRONTEND_DIST", BACKEND_DIR.parent / "frontend" / "dist"),
        run_command=shlex.split(os.environ.get("LCA_RUN_COMMAND", "")),
        read_repo_results=os.environ.get("LCA_READ_REPO_RESULTS", "1") not in {"0", "false", "no"},
        cors_origins=[
            o.strip()
            for o in os.environ.get("LCA_CORS_ORIGINS", "http://localhost:5173").split(",")
            if o.strip()
        ],
    )
