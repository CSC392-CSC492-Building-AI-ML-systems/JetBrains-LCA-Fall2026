"""How much room is there for a context editor on the 24 development datapoints (EXPERIMENT_LOG 8.106)? Uses the E4 lexical failure split of RAG F
(figures: a wrong line is a 'retrieval miss' if a name it needs from another file is not in RAG F's prompt) for the 1,372 dev lines, and splits
it for the lines whose context each stronger-agent arm changed. Dev datapoints are matched to E4 datapoints by repo and by their set of
(line number, ground truth) pairs, as in check_d0_vs_e4.py. Writes experiments/reports/E5v3/dev_headroom.md.
Usage: python experiments/scripts/report_analysis/check_headroom_dev.py"""

import gzip
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from analyze_e5_full import load  # noqa: E402

RR = os.path.join(HERE, "..", "..")


def signature(rows, dp):
    r = [v for k, v in rows.items() if k[0] == dp]
    return (r[0]["repo"], tuple(sorted((v["line_num"], v["ground_truth"]) for v in r)))


def main():
    e4 = load(os.path.join(RR, "results", "agent", "E5_full", "arms"), "e4_F_N1753")
    base = os.path.join(RR, "results", "agent", "E5v3_dev", "arms")
    dev = {k: load(base, f"e5v3dev_{k}") for k in ("d0", "d1", "d2", "d3", "d6")}
    keys = sorted(dev["d0"])
    sigs = {}
    for dp in sorted({k[0] for k in e4}):
        sigs.setdefault(signature(e4, dp), []).append(dp)
    mapping = {}
    for dp in sorted({k[0] for k in keys}):
        found = sigs.get(signature(dev["d0"], dp), [])
        assert len(found) == 1, f"dev datapoint {dp} matches {len(found)} E4 datapoints"
        mapping[dp] = found[0]
    split = {}
    with gzip.open(
        os.path.join(RR, "results", "agent", "E5_full", "arms", "e4_F_N1753", "failure_split.jsonl.gz"),
        "rt",
        encoding="utf-8",
    ) as f:
        for raw in f:
            r = json.loads(raw)
            if r["arm"] == "e4_F_N1753":
                split[(r["dp"], r["line_num"])] = r["cls"]
    cls = {k: split[(mapping[k[0]], k[1])] for k in keys}
    total = Counter(cls.values())
    wrong = sum(v for c, v in total.items() if not c.startswith("right"))
    out = [
        f"# Room for a context editor on the 24 dev datapoints ({len(keys)} lines)\n",
        "Classes of RAG F's lines at N = 1,753 from the E4 lexical failure split, matched to the dev lines (`check_headroom_dev.py`).\n",
        "| class of the RAG F line | lines | share of the 1,372 |",
        "|---|---:|---:|",
    ]
    for c, v in sorted(total.items(), key=lambda x: -x[1]):
        out.append(f"| {c} | {v} | {100 * v / len(keys):.1f}% |")
    miss = total.get("retrieval", 0)
    out += [
        f"\nWrong lines of RAG F on the dev set: {wrong} ({100 * wrong / len(keys):.1f}%); **retrieval misses (a needed name is not in the prompt): {miss} "
        f"({100 * miss / len(keys):.1f}%)**. A context editor can only add a missing name on those lines; on all 8,676 lines the share is "
        "2.0 % (171 lines).\n",
        "## Lines whose context each arm changed, by the class of the RAG F line\n",
        "| arm | changed lines | RAG F right | RAG F wrong: retrieval miss | RAG F wrong: names visible (reasoning) | RAG F wrong: self-contained |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, label in (
        ("d1", "7B planner, E5 tools"),
        ("d2", "32B planner, E5 tools"),
        ("d3", "32B planner, v3 with gate"),
        ("d6", "32B planner, v3 no gate"),
    ):
        changed = [x for x in keys if dev[k][x]["retrieval"].get("changed")]
        c = Counter(cls[x] for x in changed)
        ok = sum(v for name, v in c.items() if name.startswith("right"))
        out.append(
            f"| {label} | {len(changed)} | {ok} | {c.get('retrieval', 0)} | {c.get('reasoning', 0)} | {c.get('wrong_self_contained', 0)} |"
        )
    out += [
        f"\nReading: only {miss} of the {len(keys)} dev lines ({100 * miss / len(keys):.1f}%) are retrieval misses, the case a planner that adds a missing name "
        "can fix. The planners that edit touch few of them (7B 11, 32B with the E5 tools 8, v3 without the gate 2, v3 with the gate 0 of the 21), and "
        "most lines they change are lines where RAG F was already right or the line needs nothing from other files (7B 86 %, 32B with the E5 tools 69 %, "
        "v3 without the gate 75 %). Class names are those of `experiments/reports/E4/failure_split.md`; 'right' lines have all needed names visible."
    ]
    path = os.path.join(RR, "reports", "E5v3", "dev_headroom.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
