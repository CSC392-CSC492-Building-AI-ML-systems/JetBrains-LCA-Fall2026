import json
import os
import sys

home = os.path.expanduser("~")
a_name, b_name = sys.argv[1], sys.argv[2]
for ds in ("small_context", "medium_context"):
    base = f"{home}/lca_artifacts/python/starcoderbase-1b/{ds}/arms"

    def rows(arm):
        return [r for r in map(json.loads, open(f"{base}/{arm}/generation_results.jsonl")) if not r["zero_context"]]

    a, b = rows(a_name), rows(b_name)
    same_order = [r["ground_truth"] for r in a] == [r["ground_truth"] for r in b]
    diff = sum(x["prediction_line"] != y["prediction_line"] for x, y in zip(a, b))
    s = json.load(open(f"{base}/{b_name}/summary.json"))
    print(f"{ds}: {a_name} vs {b_name}: lines {len(a)}/{len(b)}, same order {same_order}, differing predictions {diff}, "
          f"code {s['code']}")
