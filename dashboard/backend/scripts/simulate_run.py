"""A stand-in runner for trying the launch flow without a model.

    LCA_RUN_COMMAND="python dashboard/backend/scripts/simulate_run.py" uvicorn app.main:app

It walks the run's stages, prints console lines and writes a random score.
Replace LCA_RUN_COMMAND with your real evaluation entry point; it receives the
same LCA_RUN_DIR / LCA_RUN_CONFIG environment variables.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path


def main() -> None:
    run_dir = Path(os.environ["LCA_RUN_DIR"])
    run_file = run_dir / "run.json"
    step_seconds = float(os.environ.get("SIMULATE_STEP_SECONDS", "2"))

    def update(**changes):
        data = json.loads(run_file.read_text())
        data.update(changes)
        tmp = run_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(run_file)
        return data

    data = json.loads(run_file.read_text())
    stages = data.get("stages", [])
    for i, stage in enumerate(stages):
        stage["status"] = "running"
        update(stages=stages)
        print(f"> {stage['name'].lower()}", flush=True)
        time.sleep(step_seconds)
        if stage["name"].startswith("Retrieve"):
            k = data.get("settings", {}).get("top_k", 8)
            update(evidence=[{"path": f"src/module_{n}.py", "kind": "retrieved"} for n in range(min(k, 5))])
            print(f"> retrieved {k} relevant chunks", flush=True)
        stage["status"] = "completed"
        update(stages=stages)

    score = round(random.uniform(0.3, 0.7), 2)
    update(status="completed", metrics={"score": score}, cost_usd=round(random.uniform(0.05, 0.25), 2))
    print(f"> evaluator finished: score={score}", flush=True)


if __name__ == "__main__":
    main()
