"""Cost, latency and size metrics that every context approach (static composer, RAG, agent) logs in the same way.

Per line, `LineGeneratorHF.generate_line` adds a `metrics` dict to each row of `generation_results.jsonl`
(`LINE_METRIC_FIELDS`). Per arm, `summarize` condenses those rows into the `cost` block of `summary.json`.
Nothing here changes what is generated; it only records what happened.
"""
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

LINE_METRIC_FIELDS = {
    "input_tokens": "tokens the model read for this line (context + prefix, after any window cropping)",
    "output_tokens": "tokens the model generated for this line (the stop token included)",
    "context_ms": "milliseconds the context provider spent building the prompt (0 for zero context)",
    "gen_ms": "milliseconds of model.generate (prefill + decoding), measured with the GPU synchronized",
    "truncated": "True when the prompt exceeded the model window and its beginning was cut",
    "hit_max_new_tokens": "True when generation stopped because it reached max_new_tokens, not at a newline",
}

# Fields an agent approach should add on top (not produced by the static/RAG pipeline):
AGENT_EXTRA_FIELDS = {
    "steps": "number of reasoning/search steps taken before answering",
    "tool_calls": "number of tool calls (for example repository searches or file reads)",
    "total_tokens": "tokens read plus generated across all model calls for this line",
}


def percentile(values: list[float], q: float) -> Optional[float]:
    return round(float(np.percentile(values, q)), 2) if values else None


def summarize(results_path: str, wall_seconds: Optional[float] = None) -> dict:
    """Aggregate the per-line `metrics` of one arm (rows without metrics are ignored)."""
    rows = []
    for line in Path(results_path).read_text().splitlines():
        row = json.loads(line)
        if "metrics" in row:
            rows.append(row["metrics"])
    if not rows:
        return {"lines_with_metrics": 0}

    def col(name):
        return [float(r[name]) for r in rows if r.get(name) is not None]

    summary = {
        "lines_with_metrics": len(rows),
        "input_tokens_total": int(sum(col("input_tokens"))),
        "input_tokens_mean": round(float(np.mean(col("input_tokens"))), 1),
        "output_tokens_total": int(sum(col("output_tokens"))),
        "output_tokens_mean": round(float(np.mean(col("output_tokens"))), 2),
        "context_ms_mean": round(float(np.mean(col("context_ms"))), 2),
        "context_ms_p50": percentile(col("context_ms"), 50),
        "context_ms_p95": percentile(col("context_ms"), 95),
        "gen_ms_mean": round(float(np.mean(col("gen_ms"))), 2),
        "gen_ms_p50": percentile(col("gen_ms"), 50),
        "gen_ms_p95": percentile(col("gen_ms"), 95),
        "share_truncated": round(float(np.mean([bool(r.get("truncated")) for r in rows])), 4),
        "share_hit_max_new_tokens": round(float(np.mean([bool(r.get("hit_max_new_tokens")) for r in rows])), 4),
    }
    if wall_seconds is not None:
        summary["wall_seconds_per_line"] = round(wall_seconds / len(rows), 3)
    return summary


def cost_block(results_path: str, wall_seconds: float, gpu_name: Optional[str], peak_gpu_gb: Optional[float]) -> dict:
    """The `cost` entry of summary.json: line metrics plus hardware and money.

    The price per GPU-hour comes from the environment variable LCA_USD_PER_HOUR (set it to the hourly rate of a
    rented GPU; leave it unset for a local GPU, where no dollar cost is claimed).
    """
    block = summarize(results_path, wall_seconds)
    block["gpu"] = gpu_name
    block["peak_gpu_memory_gb"] = peak_gpu_gb
    rate = os.environ.get("LCA_USD_PER_HOUR")
    block["usd_per_gpu_hour"] = float(rate) if rate else None
    block["arm_cost_usd"] = round(wall_seconds / 3600 * float(rate), 3) if rate else None
    return block


def crop_to_last_tokens(text: str, max_tokens: int, tokenizer) -> str:
    """Keep the last `max_tokens` tokens of `text`, cutting at a token boundary (the original text is not re-decoded).

    This is how the benchmark's own pipeline limits the context size N: it keeps the last N context tokens
    (`model_hub/model_inference.py`). Needs a tokenizer that returns `offset_mapping` (any fast tokenizer).
    """
    if max_tokens <= 0:
        return ""
    window = text[-max_tokens * 8:]  # a token is rarely shorter than one character in eight: bounds the work
    encoded = tokenizer(window, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if len(offsets) <= max_tokens:
        return window
    return window[offsets[-max_tokens][0]:]
