import json
import os

base = os.path.expanduser("~/lca_artifacts/python/starcoderbase-1b/small_context/arms")


def rows(arm):
    path = os.path.join(base, arm, "generation_results.jsonl")
    return [r for r in map(json.loads, open(path)) if not r["zero_context"]]


a, b = rows("r0_path_distance"), rows("v1_path_distance_provider")
print("lines:", len(a), len(b))
print("same ground-truth order:", [r["ground_truth"] for r in a] == [r["ground_truth"] for r in b])
diff = [i for i, (x, y) in enumerate(zip(a, b)) if x["prediction_line"] != y["prediction_line"]]
print("differing predictions:", len(diff))
for i in diff[:5]:
    print("  ", repr(a[i]["prediction_line"][:70]), "|", repr(b[i]["prediction_line"][:70]))
s = json.load(open(os.path.join(base, "v1_path_distance_provider", "summary.json")))
print("summary:", s["static_mode"], s["code"], "wall", s["wall_seconds"], "s; EM all", round(s["scores"]["em"]["all"], 4))
s0 = json.load(open(os.path.join(base, "r0_path_distance", "summary.json")))
print("R0 path_distance EM all:", round(s0["scores"]["em"]["all"], 4), "wall", s0["wall_seconds"], "s")
