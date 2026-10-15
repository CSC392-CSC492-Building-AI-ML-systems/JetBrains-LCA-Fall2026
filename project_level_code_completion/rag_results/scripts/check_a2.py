import json
import os
from collections import defaultdict

base = os.path.expanduser("~/lca_artifacts/python/starcoderbase-1b/small_context/arms")


def per_dp(arm):
    stats = defaultdict(lambda: [0, 0, ""])  # lines, lines with no chunk
    for r in map(json.loads, open(os.path.join(base, arm, "generation_results.jsonl"))):
        if r["zero_context"] or "retrieval" not in r:
            continue
        s = stats[r["datapoint_idx"]]
        s[0] += 1
        s[1] += not r["retrieval"]["paths"]
        s[2] = r["repo"]
    return stats


default, a2 = per_dp("dev_rag"), per_dp("dev_rag_A2")
print(f"{'dp':>3s} {'repo':42s}{'lines':>6s}{'no-chunk E1-default':>21s}{'no-chunk A2 (py only)':>23s}")
for dp in sorted(a2):
    n, z2, repo = a2[dp]
    z1 = default[dp][1]
    flag = "  <-- all empty without non-py files" if z2 == n and z1 < n else ""
    print(f"{dp:3d} {repo[:42]:42s}{n:6d}{z1:21d}{z2:23d}{flag}")
all_empty = [dp for dp in a2 if a2[dp][1] == a2[dp][0]]
print(f"\ndatapoints whose lines ALL retrieved nothing under A2: {len(all_empty)} of {len(a2)}; "
      f"they account for {sum(a2[d][0] for d in all_empty)} of {sum(v[0] for v in a2.values())} lines")
