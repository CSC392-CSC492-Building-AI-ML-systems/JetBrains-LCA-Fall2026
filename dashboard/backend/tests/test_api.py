from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.main import create_app  # noqa: E402
from app.settings import load_settings  # noqa: E402


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "baseline.yaml").write_text("task: bug_localization\n")
    (tmp_path / "bug_localization" / "configs").mkdir(parents=True)
    (tmp_path / "bug_localization" / "configs" / "run.yaml").write_text("data_path: x\n")
    return tmp_path


def make_client(repo: Path, **overrides) -> TestClient:
    settings = replace(
        load_settings(),
        repo_root=repo,
        results_dir=repo / "results",
        configs_dir=repo / "configs",
        frontend_dist=repo / "no-dist",
        **overrides,
    )
    return TestClient(create_app(settings))


def write_run(results: Path, run_id: str, **fields) -> Path:
    d = results / run_id
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"id": run_id, **fields}))
    return d


def test_reads_task_result_folders(repo):
    lib = repo / "library_based_code_generation" / "results"
    for sub, recall in (("gpt-x", 0.30), ("bm25/gpt-x", 0.36), ("other/model-y", 0.10)):
        d = lib / sub
        d.mkdir(parents=True)
        (d / "metadata.json").write_text(json.dumps({"metrics": {"ChrF": {"mean": 0.4}, "API_recall": {"mean": recall}}}))
        for i in range(3):
            (d / f"{i}.py").write_text("print(1)\n")

    ci = repo / "ci-builds-repair" / "bench" / "out"
    ci.mkdir(parents=True)
    jobs = [{"repo_name": "r", "id": i, "difficulty": 1 + i % 2, "branch_name": f"u__diff__id_{i}",
             "conclusion": "success" if i < 3 else "failure", "url": f"https://example.test/{i}"} for i in range(4)]
    (ci / "jobs_results.jsonl").write_text("\n".join(json.dumps(j) for j in jobs))
    (ci / "jobs_ids.jsonl").write_text("\n".join(json.dumps(j) for j in jobs))
    (ci / "jobs_invalid.jsonl").write_text(json.dumps({"repo_name": "bad", "id": 9}))

    ms = repo / "module_summarization" / "data" / "full" / "preds"
    ms.mkdir(parents=True)
    (ms / "summary.json").write_text(json.dumps({"model": "gpt-m", "num_predictions": 5, "num_dataset_rows": 5,
                                                 "max_context_toks": 2000, "rouge_l_f1": {"mean": 0.2}}))
    (ms.parent / "compscore_results.json").write_text(json.dumps(
        {"prediction_dir": "./data/full/preds", "compscore_mean": 0.61, "num_examples": 5}))

    client = make_client(repo)
    runs = {r["id"]: r for r in client.get("/api/runs").json()}
    assert runs["lbcg-bm25-gpt-x"]["approach"] == "rag"
    assert runs["lbcg-gpt-x"]["approach"] == "baseline"
    assert runs["ms-full-preds"]["score"] == pytest.approx(0.61)

    ci_run = client.get("/api/runs/ci-bench-out").json()
    assert ci_run["score"] == pytest.approx(0.75)
    assert ci_run["model"] == "fix_apply_diff (oracle)"
    assert ci_run["failure_count"] == 2  # one failed build + one invalid job
    assert client.get("/api/runs/ci-bench-out/log").json()["lines"]

    report = client.get("/api/compare", params={"task": "library_based_code_generation"}).json()
    assert report["model"] == "gpt-x"  # covers the most approaches
    assert {r["approach"]: r["delta_vs_baseline"] for r in report["rows"]}["rag"] == pytest.approx(0.06)
    assert [row["model"] for row in report["leaderboard"]] == ["gpt-x", "other/model-y"]
    other = client.get("/api/compare", params={"task": "library_based_code_generation", "model": "other/model-y"}).json()
    assert [r["approach"] for r in other["rows"]] == ["baseline"]

    sources = {s["id"]: s["runs"] for s in client.get("/api/meta").json()["sources"]}
    assert sources == {"dashboard": 0, "library_based_code_generation": 3, "ci-builds-repair": 1,
                       "module_summarization": 1}

def test_reads_repo_results_and_compares(repo):
    results = repo / "results"
    write_run(results, "b", task="bug_localization", approach="baseline", metrics={"score": 0.4},
              created_at="2026-10-01T00:00:00+00:00")
    d = write_run(results, "r", task="bug_localization", approach="rag", metrics={"score": 0.55},
                  created_at="2026-10-02T00:00:00+00:00", observation="better")
    (d / "failures.jsonl").write_text(
        json.dumps({"instance_id": "x1", "category": "reasoning", "summary": "wrong"}) + "\n"
    )
    write_run(results, "old", task="bug_localization", approach="rag", metrics={"score": 0.1},
              created_at="2026-09-01T00:00:00+00:00")

    client = make_client(repo)

    report = client.get("/api/compare", params={"task": "bug_localization"}).json()
    rows = {r["approach"]: r for r in report["rows"]}
    assert [r["approach"] for r in report["rows"]] == ["baseline", "rag"]
    assert rows["rag"]["run_id"] == "r"  # newest completed run wins
    assert rows["rag"]["delta_vs_baseline"] == pytest.approx(0.15)
    assert report["example_finding"]["instance_id"] == "x1"

    failures = client.get("/api/failures", params={"approach": "rag"}).json()
    assert failures["counts"]["reasoning"] == 1
    assert failures["items"][0]["run_id"] == "r"


def test_run_detail_log_and_404(repo):
    d = write_run(repo / "results", "r1", task="bug_localization", approach="rag", status="running")
    (d / "log.txt").write_text("a\nb\nc\n")
    client = make_client(repo)
    assert client.get("/api/runs/r1").json()["status"] == "running"
    assert client.get("/api/runs/r1/log", params={"offset": 1}).json() == {"lines": ["b", "c"], "next_offset": 3}
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/..%2Fetc").status_code == 404


def test_launch_without_runner_queues(repo):
    client = make_client(repo, run_command=[])
    res = client.post("/api/runs", json={"task": "bug_localization", "approach": "rag", "model": "m",
                                         "settings": {"top_k": 8}})
    assert res.status_code == 201
    run = res.json()
    assert run["status"] == "queued"
    assert "top_k: 8" in run["config_yaml"]
    assert [s["name"] for s in run["stages"]][1] == "Retrieve context"


def test_launch_with_simulated_runner(repo, monkeypatch):
    monkeypatch.setenv("SIMULATE_STEP_SECONDS", "0")
    script = BACKEND / "scripts" / "simulate_run.py"
    client = make_client(repo, run_command=[sys.executable, str(script)])
    run_id = client.post("/api/runs", json={"task": "bug_localization", "approach": "agent", "model": "m"}).json()["id"]
    for _ in range(100):
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert run["status"] == "completed"
    assert run["score"] is not None
    assert all(s["status"] == "completed" for s in run["stages"])
    assert "exited with code 0" in client.get(f"/api/runs/{run_id}/log").json()["lines"][-1]


def test_rejects_unknown_task(repo):
    client = make_client(repo)
    assert client.post("/api/runs", json={"task": "x", "approach": "rag", "model": "m"}).status_code == 422


def test_save_draft_and_list_configs(repo):
    client = make_client(repo)
    res = client.post("/api/configs/drafts", json={"task": "bug_localization", "approach": "rag", "model": "m"})
    assert res.status_code == 201
    configs = {c["path"]: c for c in client.get("/api/configs").json()}
    assert "configs/baseline.yaml" in configs
    assert "configs/drafts/rag-bug-localization.yaml" in configs
    assert configs["bug_localization/configs/run.yaml"]["task"] == "bug_localization"
    tree = client.get("/api/project-tree").json()
    assert {n["name"] for n in tree} == {"configs", "bug_localization"}
