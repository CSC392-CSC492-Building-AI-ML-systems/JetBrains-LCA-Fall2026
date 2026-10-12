"""API schemas. ``RunRecord`` mirrors the ``run.json`` file written for every
experiment (see dashboard/README.md for the full on-disk format)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RunStatus = Literal["queued", "running", "completed", "failed"]
StageStatus = Literal["waiting", "running", "completed", "failed", "skipped"]


class Option(BaseModel):
    id: str
    label: str
    description: str = ""


class TaskInfo(BaseModel):
    id: str
    label: str
    metric: str
    higher_is_better: bool = True


class Stage(BaseModel):
    name: str
    status: StageStatus = "waiting"
    detail: str | None = None


class Evidence(BaseModel):
    path: str
    # retrieved: supplied to the model; expected: in the gold answer; both: hit
    kind: Literal["retrieved", "expected", "both"] = "retrieved"
    score: float | None = None
    note: str | None = None


class TokenUsage(BaseModel):
    input: int | None = None
    output: int | None = None


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str | None = None
    task: str
    approach: str
    model: str | None = None
    repo_version: str | None = None
    status: RunStatus = "completed"
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    runtime_s: float | None = None
    cost_usd: float | None = None
    tokens: TokenUsage | None = None
    instances: int | None = None
    observation: str | None = None
    stages: list[Stage] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    # "dashboard" for runs in results/, otherwise the repo source it was read from
    source: str = "dashboard"
    source_path: str | None = None

    @property
    def score(self) -> float | None:
        return self.metrics.get("score")


class RunSummary(BaseModel):
    id: str
    name: str
    task: str
    task_label: str
    approach: str
    approach_label: str
    model: str | None
    status: RunStatus
    created_at: str | None
    score: float | None
    runtime_s: float | None
    cost_usd: float | None
    failure_count: int
    source: str


class RunDetail(RunSummary):
    repo_version: str | None
    started_at: str | None
    finished_at: str | None
    settings: dict[str, Any]
    metrics: dict[str, float | None]
    tokens: TokenUsage | None
    instances: int | None
    observation: str | None
    stages: list[Stage]
    evidence: list[Evidence]
    config_yaml: str | None
    source_path: str | None


class LogChunk(BaseModel):
    lines: list[str]
    next_offset: int


class Failure(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    task: str
    approach: str
    instance_id: str
    category: str
    summary: str
    retrieved: list[str] = Field(default_factory=list)
    expected: list[str] = Field(default_factory=list)
    url: str | None = None


class FailureReport(BaseModel):
    categories: list[Option]
    counts: dict[str, int]
    counts_by_approach: dict[str, dict[str, int]]
    items: list[Failure]


class CompareRow(BaseModel):
    approach: str
    approach_label: str
    run_id: str
    model: str | None
    score: float | None
    delta_vs_baseline: float | None
    runtime_s: float | None
    cost_usd: float | None
    observation: str | None
    failure_count: int


class LeaderboardCell(BaseModel):
    run_id: str
    score: float | None


class LeaderboardRow(BaseModel):
    model: str
    cells: dict[str, LeaderboardCell]


class CompareReport(BaseModel):
    task: str
    task_label: str
    metric: str
    models: list[str]
    model: str | None
    rows: list[CompareRow]
    leaderboard: list[LeaderboardRow]
    example_finding: Failure | None


class SourceInfo(BaseModel):
    id: str
    label: str
    path: str
    runs: int


class Meta(BaseModel):
    tasks: list[TaskInfo]
    approaches: list[Option]
    failure_categories: list[Option]
    results_dir: str
    sources: list[SourceInfo]
    runner_configured: bool


class TreeNode(BaseModel):
    name: str
    path: str
    type: Literal["dir", "file"]
    children: list["TreeNode"] | None = None


class ConfigFile(BaseModel):
    name: str
    path: str
    content: str
    # inferred from the task folder the file lives in
    task: str | None = None


class LaunchRequest(BaseModel):
    task: str
    approach: str
    model: str = Field(min_length=1)
    repo_version: str = "main"
    name: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class DraftRequest(LaunchRequest):
    filename: str | None = None
