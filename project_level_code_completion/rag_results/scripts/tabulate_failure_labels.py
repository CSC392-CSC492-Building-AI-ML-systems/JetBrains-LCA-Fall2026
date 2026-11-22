"""Failure taxonomy of week 8 (RAG_EXPERIMENT_LOG 8.106): labels for the 125 sampled lines of `sample_failures_week8.py`, one letter per sampled item,
in the order of failure_samples_week8.txt (items 1 to 25 of each group). One reader (the assistant), subjective labels, 25 lines per group, so
every share carries roughly +-10 points of sampling noise (a 95 % interval for a share of 40 % on 25 lines is about 21 % to 61 %).
Categories as in 8.77, applied to the prediction of the arm named in the group:
  A wrong name        a different attribute, method, class, import, constant, key or variable than the repository defines
  B different statement   plausible code (or a comment or an empty line) that is not the next statement of the file
  C near miss         right idea, small deviation: missing or extra argument, wrong literal or default, off-by-one, missing wrapper
                      (`return`, `self.`, a call), quote or spacing style, a wrong word in a message
  D equivalent alternative   a valid variable name or expression that differs from the ground truth (counts as wrong under exact match)
  E free text         comment or docstring wording where the truth is a comment or docstring
  F data / fragment   numbers or ids, or a cut-off fragment
For the group `broke:agent` the label describes the agent's wrong prediction, on lines where RAG F was right.
Writes rag_results/reports/final/failure_taxonomy_week8.md. Usage: python rag_results/scripts/tabulate_failure_labels.py"""
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {
    "wrong:zero": "BCDCBBBCBDCCEADBCABCBCCBB",
    "wrong:pd": "BDBCBDBEFCBEBBCBBBBCCBBBB",
    "wrong:F": "CBFBBBDBBACCBCCBCCBBBBCBC",
    "wrong:agent": "DBCACCEBCCBBCCCBCBCAACAAB",
    "broke:agent": "BBBBBACCCABBAFCABABEBAEFD",
}
POOL = {"wrong:zero": 5432, "wrong:pd": 5232, "wrong:F": 4153, "wrong:agent": 4255, "broke:agent": 136}
TITLE = {"wrong:zero": "no context, wrong lines", "wrong:pd": "path_distance (N=1,753), wrong lines", "wrong:F": "RAG F (N=1,753), wrong lines",
         "wrong:agent": "7B planner agent, wrong lines", "broke:agent": "agent breaks a line RAG F got right (136 lines)"}
CATS = [("A", "wrong name"), ("B", "different statement"), ("C", "near miss"), ("D", "equivalent alternative"), ("E", "free text"), ("F", "data / fragment")]


def main():
    for k, v in LABELS.items():
        assert len(v) == 25 and set(v) <= set("ABCDEF"), k
    out = ["# Failure taxonomy (week 8): hand-read samples of wrong lines, medium set, CodeLlama-7B, N = 1,753\n",
           "One reader (the assistant), subjective labels, 25 random wrong lines per group (seed 0, `sample_failures_week8.py`; every line is in "
           "`failure_samples_week8.txt` with the truth and all four predictions). With 25 lines a share has roughly +-10 points of sampling noise, so "
           "only large differences mean anything. Definitions in `tabulate_failure_labels.py`.\n",
           "| group | lines in group | " + " | ".join(f"{c} {n}" for c, n in CATS) + " |", "|---|---:|" + "---:|" * len(CATS)]
    for g, s in LABELS.items():
        c = Counter(s)
        out.append(f"| {TITLE[g]} | {POOL[g]:,} | " + " | ".join(f"{c[x]} ({100 * c[x] / 25:.0f}%)" for x, _ in CATS) + " |")
    pooled = Counter("".join(LABELS[g] for g in ("wrong:zero", "wrong:pd", "wrong:F", "wrong:agent")))
    out += ["\nAll four wrong-line groups pooled (100 lines): " + ", ".join(f"{x} {n}%" for x, n in [(c, pooled[c]) for c, _ in CATS]) + ".\n",
            "## What the samples show\n",
            "- **Most failures are plausible code that is not what the author wrote next (B) or a small deviation from it (C).** Together they are "
            f"{pooled['B'] + pooled['C']} of 100 wrong lines across the four arms. Adding retrieval or an agent does not change this much: the shares of "
            "B and C stay in the same range (72 to 88 % together, per arm) for every arm, within the sampling noise.",
            "- **Wrong names (A) are the failure retrieval can fix**, and they are a minority (about 1 in 10 of wrong lines, a little higher for the agent's "
            "sample). The agent's wrong lines contain a few more wrong names than RAG F's (5 against 1 of 25), in line with the E5 finding that its added code "
            "does not supply missing names; at this sample size that difference is a hint, not a result.",
            "- **When the agent breaks a line RAG F got right**, the agent's replacement is mostly a different statement (10 of 25) or a wrong name (6 of 25): "
            "the line is not nudged slightly, it goes somewhere else. Only 4 of 25 are near misses. This fits the view that the added chunks steer the writer "
            "away from the right statement.",
            "- **A share of the wrong lines are not developer errors under exact match:** equivalent alternatives (D), comment or docstring wording (E) and "
            "fragments (F) make up about 13 % of the wrong lines of the four arms; near misses (C) would earn partial credit under edit similarity.",
            "- **The arms often fail in the same way:** on many of the sampled lines two or more arms give the identical wrong prediction "
            "(see the samples file), consistent with failures being mainly about the writer model and the open-ended next-step choice.\n",
            "Limits: one labeler, no inter-rater check, 25 lines per group, random sample of all wrong lines (no stratification by line class). The "
            "retrieval-miss analysis (names missing from the prompt) is the lexical screen of E4 and E5, not this table."]
    path = os.path.join(HERE, "..", "reports", "final", "failure_taxonomy_week8.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
