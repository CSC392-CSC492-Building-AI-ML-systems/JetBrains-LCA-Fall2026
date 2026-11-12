"""Failure split of the planner agent next to RAG F (RAG_EXPERIMENT_LOG section 8.91).

`eval/failure_split.py` puts every line into right / wrong-but-self-contained / retrieval miss (a name the line needs from another
file is not visible in the prompt) / reasoning (all needed names visible). This script lines the agent up against RAG F at
N=1,753 line by line and asks what the planner's edits did to the visibility of the needed names:

    python rag_results/scripts/analyze_agent_split.py [--base DIR] [--out report.md]

DIR (default rag_results/agent/E5_full/arms) holds e4_F_N1753/failure_split.jsonl.gz and e5_agent_full/failure_split.jsonl.gz
(made with eval.failure_split) and the arms' generation_results.jsonl.gz (for the `changed` flag). The screen is lexical (a name can
be visible without its definition being useful), so it is a screen, not a proof.
"""
import argparse
import gzip
import json
import os
from collections import Counter


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf8") as f:
        return [json.loads(line) for line in f]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=os.path.join(here, "..", "agent", "E5_full", "arms"))
    parser.add_argument("--out", default=os.path.join(here, "..", "reports", "E5", "agent_failure_split.md"))
    args = parser.parse_args()
    base = os.path.expanduser(args.base)
    rag = {(r["dp"], r["line_num"]): r for r in read_jsonl(os.path.join(base, "e4_F_N1753", "failure_split.jsonl.gz"))}
    agent_rows = read_jsonl(os.path.join(base, "e5_agent_full", "failure_split.jsonl.gz"))
    agent = {(r["dp"], r["line_num"]): r for r in agent_rows}
    assert set(rag) == set(agent), "the two arms cover different lines"
    changed = {(r["datapoint_idx"], r["line_num"]): r["retrieval"]["changed"]
               for r in read_jsonl(os.path.join(base, "e5_agent_full", "generation_results.jsonl.gz"))}
    keys = sorted(agent)
    out = []
    w = out.append

    ok = sum(r["replay_matches_logged"] for r in agent_rows)
    tok = sum(r["rebuilt_tokens"] + 1 == r["logged_tokens"] for r in agent_rows)
    w(f"# Failure split of the agent next to RAG F at N=1,753 ({len(keys)} lines)\n")
    w(f"Rebuild check for the agent arm (planner replayed from the logged actions, no GPU): the replayed final context equals the logged "
      f"one on {ok} of {len(agent_rows)} lines; the rebuilt prompt has the logged token count on {tok} of {len(agent_rows)}.\n")

    w("## 1. Classes (as in the E4 failure split)\n")
    w("| arm | lines | right | wrong, self-contained | wrong, retrieval miss | wrong, reasoning (names visible) |")
    w("|---|---:|---:|---:|---:|---:|")
    for name, rows in (("RAG F", rag), ("agent (v2)", agent)):
        c = Counter(r["cls"] for r in rows.values())
        n = sum(c.values())
        right = c["right"] + c["right_self_contained"]
        w(f"| {name} | {n} | {right} ({right / n:.1%}) | {c['wrong_self_contained']} ({c['wrong_self_contained'] / n:.1%}) | "
          f"{c['retrieval']} ({c['retrieval'] / n:.1%}) | {c['reasoning']} ({c['reasoning'] / n:.1%}) |")
    w("")

    needs = [k for k in keys if rag[k]["n_needed"] > 0]
    gained = [k for k in needs if agent[k]["n_visible"] > rag[k]["n_visible"]]
    lost = [k for k in needs if agent[k]["n_visible"] < rag[k]["n_visible"]]
    w("## 2. What the planner's edits did to the visibility of the names a line needs from other files\n")
    w(f"Lines that need at least one name from another file: {len(needs)}. Of them the planner changed the context on "
      f"{sum(changed[k] for k in needs)}.\n")
    w(f"- more needed names visible than under RAG F: **{len(gained)}** lines; fewer visible (an evicted chunk held a name): **{len(lost)}** lines; "
      f"same number: {len(needs) - len(gained) - len(lost)}")
    w("- only lines whose context was changed can differ; the counts above are over all lines that need names.\n")
    w("| | lines | right under RAG F | right under agent | only agent right | only RAG F right |")
    w("|---|---:|---:|---:|---:|---:|")
    for label, group in (("names gained", gained), ("names lost", lost)):
        r_f = sum(rag[k]["correct"] for k in group)
        r_a = sum(agent[k]["correct"] for k in group)
        only_a = sum(agent[k]["correct"] and not rag[k]["correct"] for k in group)
        only_r = sum(rag[k]["correct"] and not agent[k]["correct"] for k in group)
        w(f"| {label} | {len(group)} | {r_f} | {r_a} | {only_a} | {only_r} |")
    w("")

    w("## 3. Retrieval misses of RAG F and what happened to them\n")
    miss = [k for k in keys if rag[k]["cls"] == "retrieval"]
    now_visible = [k for k in miss if agent[k]["cls"] != "retrieval"]
    fixed = [k for k in miss if agent[k]["correct"]]
    w(f"- RAG F retrieval misses: {len(miss)}; under the agent a name set that is now fully visible: {len(now_visible)}; of those the line is right: "
      f"{sum(agent[k]['correct'] for k in now_visible)}; lines right under the agent in all: {len(fixed)}")
    new_miss = [k for k in keys if agent[k]["cls"] == "retrieval" and rag[k]["cls"] != "retrieval"]
    w(f"- new retrieval misses under the agent (RAG F did not have one): {len(new_miss)}")
    w(f"- net change in retrieval misses: {len([k for k in keys if agent[k]['cls'] == 'retrieval']) - len(miss):+d}\n")

    w("## 4. Changed lines by class pair (RAG F class -> agent class), lines whose context the planner changed\n")
    pairs = Counter((rag[k]["cls"], agent[k]["cls"]) for k in keys if changed[k])
    w("| RAG F | agent | lines |")
    w("|---|---|---:|")
    for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1]):
        w(f"| {a} | {b} | {n} |")

    text = "\n".join(out)
    print(text)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
