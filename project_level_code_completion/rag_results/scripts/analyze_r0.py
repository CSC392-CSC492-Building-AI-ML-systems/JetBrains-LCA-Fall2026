import json
import os

from eval.line_generators import dev_indices, subset_indices

base = os.path.expanduser("~/lca_artifacts/python/starcoderbase-1b/small_context/arms")
print("log: exit codes and wall times")
for line in open(os.path.expanduser("~/lca_r0.log"), errors="replace"):
    if line.startswith("====="):
        print("  ", line.strip()[:110])
print("  traceback in log:", "Traceback" in open(os.path.expanduser("~/lca_r0.log"), errors="replace").read())

print("\nsummary.json provenance")
for arm in ("r0_zero", "r0_path_distance", "r0_imports_first", "r0_v1"):
    s = json.load(open(os.path.join(base, arm, "summary.json")))
    print(f"  {arm:18s} split={s['split']} dev_size={s['dev_size']} dev_extra={s['dev_extra']} code={s['code']} "
          f"wall={s['wall_seconds']}s lines={sum(s['line_counts'].values())}")

new_dev = dev_indices(144, 20, 28)
old_dev = subset_indices(144, 20)
pos = {j: new_dev.index(o) for j, o in enumerate(old_dev)}  # E2 datapoint_idx -> R0 datapoint_idx
print(f"\nnew dev = {len(new_dev)} datapoints; old dev inside new dev: {all(o in new_dev for o in old_dev)}")


def rows(arm, zero):
    path = os.path.join(base, arm, "generation_results.jsonl")
    return [r for r in map(json.loads, open(path)) if bool(r["zero_context"]) == zero]


print("\nPrediction 4: exact reproduction of E2 on the 20 shared datapoints (per-line predictions)")
for e2, r0, zero in (("dev_zero", "r0_zero", True), ("dev_path_distance", "r0_path_distance", False),
                     ("dev_rag_A1", "r0_v1", False)):
    old_rows = rows(e2, zero)
    new_by_idx = {}
    for r in rows(r0, zero):
        new_by_idx.setdefault(r["datapoint_idx"], []).append(r)
    old_by_idx = {}
    for r in old_rows:
        old_by_idx.setdefault(r["datapoint_idx"], []).append(r)
    n_lines = n_diff = n_gt_mismatch = 0
    for j, old in old_by_idx.items():
        new = new_by_idx[pos[j]]
        if [r["ground_truth"] for r in old] != [r["ground_truth"] for r in new]:
            n_gt_mismatch += 1
            continue
        n_lines += len(old)
        n_diff += sum(a["prediction_line"] != b["prediction_line"] for a, b in zip(old, new))
    print(f"  {e2:18s} vs {r0:18s}: datapoints {len(old_by_idx)}, lines {n_lines}, "
          f"ground-truth order mismatches {n_gt_mismatch}, differing predictions {n_diff}")
