import json
import os
import statistics

base = os.path.expanduser("~/lca_artifacts/python/starcoderbase-1b/small_context/arms")


def rows(arm, zero):
    path = os.path.join(base, arm, "generation_results.jsonl")
    return [r for r in map(json.loads, open(path)) if bool(r["zero_context"]) == zero]


print("=== R: reproduction check (E1 pilot vs E2 dev, same arm, per-line predictions) ===")
for e1, e2, zero in (("pilot_zero", "dev_zero", True), ("pilot_path_distance", "dev_path_distance", False),
                     ("pilot_rag", "dev_rag", False)):
    a, b = rows(e1, zero), rows(e2, zero)
    same_gt = [x["ground_truth"] for x in a] == [x["ground_truth"] for x in b]
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x["prediction_line"] != y["prediction_line"]]
    print(f"{e2:20s} lines {len(a)} vs {len(b)} | same ground-truth order: {same_gt} | "
          f"differing predictions: {len(diff)}")

print("\n=== Retrieval diagnostics per RAG arm (dev, 709 lines) ===")
print(f"{'arm':12s}{'zero-chunk':>11s}{'mean chunks':>12s}{'py share':>10s}{'infile 0-chunk':>16s}")
for arm in ("dev_rag", "dev_rag_A1", "dev_rag_A2", "dev_rag_A3"):
    rs = [r for r in rows(arm, False) if "retrieval" in r]
    n0 = sum(1 for r in rs if not r["retrieval"]["paths"])
    chunks = [p for r in rs for p in r["retrieval"]["paths"]]
    py = sum(p.endswith(".py") for p in chunks) / max(1, len(chunks))
    infile = [r for r in rs if r["line_class"] == "infile"]
    n0i = sum(1 for r in infile if not r["retrieval"]["paths"])
    print(f"{arm:12s}{n0:11d}{statistics.mean(len(r['retrieval']['paths']) for r in rs):12.1f}{py:10.0%}"
          f"{n0i:>10d}/{len(infile)}")

summary = json.load(open(os.path.join(base, "dev_rag", "summary.json")))
print("\nsummary.json check: split =", summary["split"], "dev_size =", summary["dev_size"],
      "budget =", summary["token_budget"])
