import json
import os
import re
import statistics

base = os.path.expanduser("~/lca_artifacts/python/starcoderbase-1b/small_context/arms")
log = open(os.path.expanduser("~/lca_pilot.log"), errors="replace").read()
print("log mentions Traceback:", "Traceback" in log, "| 'Error' lines:",
      [l[:100] for l in log.splitlines() if re.search(r"\bError\b", l)][:3])

arms = {}
for arm in ("zero", "path_distance", "rag"):
    d = os.path.join(base, f"pilot_{arm}")
    s = json.load(open(os.path.join(d, "summary.json")))
    key_em, key_es = ("em_zero", "es_zero") if arm == "zero" else ("em", "es")
    arms[arm] = (s["scores"][key_em], s["scores"][key_es], s)
    print(f"\n[{arm}] datapoints={s['limit_datapoints']} budget={s['token_budget']} lines={s['line_counts']}")

cats = list(arms["zero"][0])
print("\nEM (exact match) per category")
print(f"{'category':16s}" + "".join(f"{a:>16s}" for a in arms))
for c in cats:
    print(f"{c:16s}" + "".join(f"{arms[a][0].get(c, float('nan')):16.3f}" for a in arms))
print("\nES (edit similarity) per category")
for c in cats:
    print(f"{c:16s}" + "".join(f"{arms[a][1].get(c, float('nan')):16.2f}" for a in arms))

print("\nEM difference vs zero (positive/negative/zero share of datapoints), all lines")
for a in ("path_distance", "rag"):
    print(f"  {a}: {arms[a][2]['em_difference_vs_zero'].get('all')}")

# Per-line analysis of the RAG arm
rows = [json.loads(l) for l in open(os.path.join(base, "pilot_rag", "generation_results.jsonl"))]
rows = [r for r in rows if not r["zero_context"]]
ret = [r["retrieval"] for r in rows if "retrieval" in r]
print(f"\nRAG per-line records: {len(rows)} | with retrieval info: {len(ret)}")
print("  fallback (empty prefix):", sum(r["fallback"] for r in ret))
print("  chunks selected: mean %.1f, median %s, zero-chunk lines: %d" % (
    statistics.mean(len(r["paths"]) for r in ret), statistics.median(len(r["paths"]) for r in ret),
    sum(1 for r in ret if not r["paths"])))
ext = {}
for r in ret:
    for p in r["paths"]:
        e = os.path.splitext(p)[1] or "(none)"
        ext[e] = ext.get(e, 0) + 1
tot = sum(ext.values())
print("  retrieved chunks by extension:", {k: f"{v/tot:.0%}" for k, v in sorted(ext.items(), key=lambda kv: -kv[1])[:6]})

# Same lines compared across arms (pilot jsonl holds both passes for zero arm)
def load(arm, zero):
    rows = [json.loads(l) for l in open(os.path.join(base, f"pilot_{arm}", "generation_results.jsonl"))]
    return [r for r in rows if bool(r["zero_context"]) == zero]
z, p, g = load("zero", True), load("path_distance", False), load("rag", False)
print(f"\nlines per arm: zero={len(z)} path_distance={len(p)} rag={len(g)}")
if len(z) == len(p) == len(g):
    em = lambda rs: [r["prediction_line"].strip() == r["ground_truth"].strip() for r in rs]
    ez, ep, eg = em(z), em(p), em(g)
    print("line-level EM: zero=%.3f path_distance=%.3f rag=%.3f" % (
        sum(ez) / len(ez), sum(ep) / len(ep), sum(eg) / len(eg)))
    print("rag right & path_distance wrong:", sum(a and not b for a, b in zip(eg, ep)),
          "| path_distance right & rag wrong:", sum(b and not a for a, b in zip(eg, ep)))
    print("rag right & zero wrong:", sum(a and not b for a, b in zip(eg, ez)),
          "| zero right & rag wrong:", sum(b and not a for a, b in zip(eg, ez)))
