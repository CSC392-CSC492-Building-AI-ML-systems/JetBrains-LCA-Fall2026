"""What the planner's edits look like in numbers: what it adds, what it displaces (RAG_EXPERIMENT_LOG section 8.92).

Compares, for every line whose context the agent changed, the chunk list of the agent arm with the one of the RAG F arm
(both at N=1,753, from the retrieval info saved with each line) and prints the counts. Plain Python, no GPU.

    python rag_results/scripts/agent_change_stats.py [--base DIR] [--out FILE]
"""
import argparse
import collections
import gzip
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))


def rows(base, arm):
    with gzip.open(os.path.join(base, arm, "generation_results.jsonl.gz"), "rt", encoding="utf8") as f:
        return {(r["datapoint_idx"], r["line_num"]): r for r in map(json.loads, f)}


def right(row):
    return row["prediction_line"].strip() == row["ground_truth"].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=os.path.join(HERE, "..", "agent", "E5_full", "arms"))
    parser.add_argument("--out", default=os.path.join(HERE, "..", "reports", "E5", "agent_change_stats.txt"))
    args = parser.parse_args()
    rag, agent = rows(args.base, "e4_F_N1753"), rows(args.base, "e5_agent_full")
    changed = [k for k in agent if agent[k]["retrieval"]["changed"]]
    out = []

    def chunks(info):
        return [(p, tuple(l)) for p, l in zip(info["paths"], info["lines"])]

    kinds_added, lengths, displaced_count, added_count = collections.Counter(), [], collections.Counter(), collections.Counter()
    same_file = displaced_total = 0
    displaced_lines = set()
    for k in changed:
        a, r = agent[k]["retrieval"], rag[k]["retrieval"]
        agent_chunks, rag_chunks = chunks(a), chunks(r)
        added = [(c, kind) for c, kind in zip(agent_chunks, a["kinds"]) if c not in rag_chunks]
        displaced = [c for c in rag_chunks if c not in agent_chunks]
        if displaced:
            displaced_lines.add(k)
        displaced_count[len(displaced)] += 1
        added_count[len(added)] += 1
        for (path, (start, end)), kind in added:
            kinds_added[kind] += 1
            lengths.append(end - start + 1)
        for path, _ in displaced:
            displaced_total += 1
            same_file += any(path == c[0] for c, _ in added)
    default_lengths = [l[1] - l[0] + 1 for k in changed for l in rag[k]["retrieval"]["lines"]]
    out.append(f"lines whose context the agent changed: {len(changed)}")
    out.append(f"lines where at least one RAG F chunk is no longer in the prompt: {len(displaced_lines)} ({100 * len(displaced_lines) / len(changed):.1f}%)")
    out.append(f"RAG F chunks displaced per changed line (count: lines): {sorted(displaced_count.items())}")
    out.append(f"chunks added per changed line (count: lines): {sorted(added_count.items())}")
    out.append(f"kinds of the added chunks: {kinds_added.most_common()}")
    out.append(f"added chunk length in lines: median {statistics.median(lengths):.0f}, mean {statistics.mean(lengths):.0f}; "
               f"RAG F's chunks on the same lines: median {statistics.median(default_lengths):.0f}, mean {statistics.mean(default_lengths):.0f}")
    out.append(f"displaced chunks that come from the same file as an added chunk: {same_file} of {displaced_total} ({100 * same_file / displaced_total:.1f}%)")
    for label, group in (("something displaced", displaced_lines), ("nothing displaced", set(changed) - displaced_lines)):
        only_agent = sum(right(agent[k]) and not right(rag[k]) for k in group)
        only_rag = sum(right(rag[k]) and not right(agent[k]) for k in group)
        out.append(f"{label}: {len(group)} lines; only the agent right {only_agent}, only RAG F right {only_rag}, net {only_agent - only_rag:+d}")
    text = "\n".join(out)
    print(text)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
