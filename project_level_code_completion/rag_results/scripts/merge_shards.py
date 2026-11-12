"""Merge the shards of one arm (RAG_EXPERIMENT_LOG section 8.87) into the folder an unsharded run would have produced.

    python rag_results/scripts/merge_shards.py OUT_DIR SHARD_DIR [SHARD_DIR ...]

Each shard folder is an arm folder written with `shard_index=i shard_count=k` (summary.json, generation_results.jsonl,
per_datapoint_scores.json, config.json). Shard i evaluated datapoints i, i+k, i+2k, ... of the whole dataset, and its saved
`datapoint_idx` is the position inside the shard, so the original index is i + position * k. The merge restores the
original index in every line, puts the per-datapoint score lists in original datapoint order (needed for paired tests with
arms that were not sharded), sums the line counts, recomputes the scores and the cost block from the merged lines, and
checks that nothing is missing or duplicated. It never edits a shard folder and refuses to write into a non-empty OUT_DIR.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.run_metrics import summarize  # noqa: E402


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def merge(out_dir: Path, shard_dirs: list[Path]) -> dict:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty; refusing to write into it")
    shards = []
    for directory in shard_dirs:
        summary = json.loads((directory / "summary.json").read_text())
        if summary.get("shard_count") is None:
            raise SystemExit(f"{directory} is not a shard (summary.json has no shard_count)")
        shards.append({"dir": directory, "summary": summary, "rows": read_rows(directory / "generation_results.jsonl"),
                       "scores": json.loads((directory / "per_datapoint_scores.json").read_text()),
                       "config": json.loads((directory / "config.json").read_text())})
    shards.sort(key=lambda s: s["summary"]["shard_index"])
    count = shards[0]["summary"]["shard_count"]
    if [s["summary"]["shard_index"] for s in shards] != list(range(count)):
        raise SystemExit(f"need exactly shards 0..{count - 1}, got {[s['summary']['shard_index'] for s in shards]}")
    same = ("arm", "dataset", "model", "token_budget", "context_cap", "split", "lines", "static_mode", "shard_count")
    for key in same:
        values = {json.dumps(s["summary"].get(key)) for s in shards}
        if len(values) != 1:
            raise SystemExit(f"shards disagree on {key}: {sorted(values)}")

    # the number of datapoints of each shard must be what `range(i, n, k)` gives for one total n
    sizes = [len(s["scores"]["em_list"]["all"]) if "em_list" in s["scores"] else len(s["scores"]["em_zero_list"]["all"])
             for s in shards]
    n_items = sum(sizes)
    for i, size in enumerate(sizes):
        if size != len(range(i, n_items, count)):
            raise SystemExit(f"shard {i} has {size} datapoints, but {n_items} datapoints split in {count} give "
                             f"{len(range(i, n_items, count))}")

    rows = []
    for s in shards:
        i = s["summary"]["shard_index"]
        positions = {r["datapoint_idx"] for r in s["rows"]}
        if not positions <= set(range(sizes[i])):
            raise SystemExit(f"shard {i} has lines for positions outside 0..{sizes[i] - 1}")
        for row in s["rows"]:
            rows.append({**row, "datapoint_idx": i + row["datapoint_idx"] * count})
    rows.sort(key=lambda r: r["datapoint_idx"])  # stable: the lines of a datapoint keep their order
    keys = [(r["datapoint_idx"], r["line_num"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("a (datapoint_idx, line_num) pair occurs twice: a shard was run twice or appended to")

    merged_scores = {}
    for list_name in shards[0]["scores"]:  # em_list, es_list (and em_zero_list, es_zero_list when present)
        merged_scores[list_name] = {}
        categories = list(dict.fromkeys(c for s in shards for c in s["scores"].get(list_name, {})))  # union, in order
        for category in categories:
            pieces = []
            for s in shards:
                i = s["summary"]["shard_index"]
                values = s["scores"].get(list_name, {}).get(category, [])
                if category == "all":
                    positions = list(range(sizes[i]))
                else:  # one entry per datapoint that has lines of this class, in datapoint order
                    positions = sorted({r["datapoint_idx"] for r in s["rows"] if r["line_class"] == category})
                if len(values) != len(positions):
                    raise SystemExit(f"shard {i}, {list_name}[{category}]: {len(values)} scores for {len(positions)} datapoints")
                pieces += [(i + p * count, v) for p, v in zip(positions, values)]
            merged_scores[list_name][category] = [v for _, v in sorted(pieces)]
    scores = {name[:-len("_list")]: {category: statistics.fmean(values) for category, values in lists.items()}
              for name, lists in merged_scores.items()}

    line_counts = {}
    for s in shards:
        for category, n in s["summary"]["line_counts"].items():
            line_counts[category] = line_counts.get(category, 0) + n
    from_rows = {}
    for r in rows:
        from_rows[r["line_class"]] = from_rows.get(r["line_class"], 0) + 1
    if line_counts != from_rows:
        raise SystemExit(f"line counts disagree: summaries {line_counts}, lines {from_rows}")

    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / "generation_results.jsonl"
    results.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    wall = round(sum(s["summary"]["wall_seconds"] for s in shards), 1)
    cost = summarize(str(results), wall)
    costs = [s["summary"]["cost"] for s in shards]
    for field in ("gpu", "usd_per_gpu_hour"):
        values = {c.get(field) for c in costs}
        cost[field] = values.pop() if len(values) == 1 else "mixed: " + ", ".join(sorted(map(str, values)))
    cost["peak_gpu_memory_gb"] = max((c.get("peak_gpu_memory_gb") or 0) for c in costs)
    cost["arm_cost_usd"] = (round(sum(c["arm_cost_usd"] for c in costs), 3)
                            if all(c.get("arm_cost_usd") is not None for c in costs) else None)
    base = shards[0]["summary"]
    summary = {**base, "arm_name": out_dir.name, "shard_index": None, "shard_count": None, "wall_seconds": wall,
               "wall_seconds_longest_shard": max(s["summary"]["wall_seconds"] for s in shards),
               "cost": cost, "scores": scores, "line_counts": line_counts,
               "merged_from": [{"dir": s["dir"].name, "shard_index": s["summary"]["shard_index"],
                                "datapoints": sizes[s["summary"]["shard_index"]], "wall_seconds": s["summary"]["wall_seconds"],
                                "code": s["summary"].get("code")} for s in shards]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=4))
    (out_dir / "per_datapoint_scores.json").write_text(json.dumps(merged_scores))
    config = {**shards[0]["config"], "shard_index": None, "shard_count": None, "merged_from_shards": count}
    (out_dir / "config.json").write_text(json.dumps(config, indent=4))
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    result = merge(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
    print(f"merged {len(result['merged_from'])} shards: {result['cost']['lines_with_metrics']} lines, "
          f"EM all {result['scores']['em']['all']:.4f}, wall {result['wall_seconds']} s summed")
