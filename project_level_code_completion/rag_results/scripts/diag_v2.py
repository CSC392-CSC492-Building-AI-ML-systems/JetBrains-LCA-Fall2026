import json
import os

home = os.path.expanduser("~")
S = f"{home}/lca_artifacts/python/starcoderbase-1b/small_context/arms"
targets = json.load(open(f"{home}/lca_artifacts/reports/targets_small_dev.json"))

all_rows = [r for r in map(json.loads, open(f"{S}/v2_zero_all/generation_results.jsonl")) if r["zero_context"]]
kept = [r for r in all_rows if r["line_num"] in targets.get(str(r["datapoint_idx"]), {}).get(r["line_class"], [])]
ref = [r for r in map(json.loads, open(f"{S}/r0t_zero/generation_results.jsonl")) if r["zero_context"]]
print("kept", len(kept), "reference", len(ref))
print("ground truth sequences identical:", [r["ground_truth"] for r in kept] == [r["ground_truth"] for r in ref])
diff = [(a, b) for a, b in zip(kept, ref) if a["prediction_line"] != b["prediction_line"]]
print("differing predictions:", len(diff))
for a, b in diff[:6]:
    print(f"  repo={a['repo'][:25]} class={a['line_class']} line={a['line_num']}")
    print("    all-lines run :", repr(a["prediction_line"][:80]))
    print("    targeted run  :", repr(b["prediction_line"][:80]))
    print("    ground truth  :", repr(a["ground_truth"][:80]))
# Is the all-lines zero run itself consistent with the earlier dev all-lines zero run (r0_zero)?
r0 = [r for r in map(json.loads, open(f"{S}/r0_zero/generation_results.jsonl")) if r["zero_context"]]
d0 = sum(a["prediction_line"] != b["prediction_line"] for a, b in zip(all_rows, r0))
print(f"v2_zero_all vs r0_zero (same lines=all setting): {len(all_rows)} vs {len(r0)} lines, differing predictions {d0}")
