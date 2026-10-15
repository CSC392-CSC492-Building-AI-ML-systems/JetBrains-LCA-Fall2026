"""Verification checks V-B, V-C, V-D (research log section 8.30). CPU only."""
import json
import math
import os
import random
import sys

import numpy as np
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer

from composers.path_distance_composer import PathDistanceComposer
from eval.analyze_arms import arm_lines, datapoint_means
from eval.line_generators import heldout_indices
from eval.preprocessors import load_datapoints
from eval.proxy_metrics import visible_prompt
from rag.builder import RagContextBuilder

HOME = os.path.expanduser("~")
ARMS = f"{HOME}/lca_artifacts/python/starcoderbase-1b"
BUDGET = 7900
F = dict(prefix_cap=7900, continuation_lines=20)
which = sys.argv[1:] or ["B", "C", "D"]

tok = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b")
encode = lambda t: tok(t, add_special_tokens=False)["input_ids"]
decode = lambda ids: tok.decode(ids)
count = lambda t: len(encode(t))


def load(dataset, dev_size, dev_extra):
    with initialize_config_dir(config_dir=os.path.join(os.environ["PROJECT"], "eval", "config"), version_base=None):
        cfg = compose(config_name="compare_arms", overrides=[f"dataset={dataset}"])
    data = load_datapoints(cfg.dataset)
    return [data[i] for i in heldout_indices(len(data), dev_size, dev_extra)]  # position p = datapoint_idx


def builder_for(dp):
    path = next(iter(dp.get_completion()))
    return RagContextBuilder(dp.get_context(), path, dp.repo_name, count, lang_sep="", meta_sep="METASEP\n"), path


medium = load("medium", 28, 28)
small = load("small", 20, 28)

# ----------------------------------------------------------------------------------------------- V-B
if "B" in which:
    print("=================== V-B: rebuilt F prompts on 400 random held-out lines ===================")
    rng = random.Random(123)
    plan = [("medium", p) for p in rng.sample(range(len(medium)), 70)] + [("small", p) for p in rng.sample(range(len(small)), 30)]
    n = over = leaks = bad_tail = 0
    max_tokens = 0
    copy_seen = 0
    for name, p in plan:
        dp = (medium if name == "medium" else small)[p]
        builder, path = builder_for(dp)
        lines = [(cls, ln) for cls, nums in dp.completion_lines.items() for ln in nums]
        for cls, ln in rng.sample(lines, min(4, len(lines))):
            prefix = dp.get_prefix(ln)
            r = builder.context_for(prefix, BUDGET, **F)
            prompt = r.context + "\n" + r.prefix
            tokens = count(prompt)
            max_tokens = max(max_tokens, tokens)
            n += 1
            over += tokens > BUDGET
            leaks += any(c.path == path for c in r.chunks)
            bad_tail += not prompt.endswith(r.prefix)
            g = dp.get_line(ln).strip()
            copy_seen += len(g) >= 15 and g in {x.strip() for x in r.context.split("\n")}
    print(f"lines checked: {n} | max prompt tokens: {max_tokens} (budget {BUDGET}) | budget violations: {over} | "
          f"chunks from the completion file: {leaks} | prompts not ending in the prefix: {bad_tail}")
    print(f"lines (>=15 chars) whose ground truth appears verbatim in the retrieved context: {copy_seen} of {n}")

# ----------------------------------------------------------------------------------------------- V-C
if "C" in which:
    print("\n=================== V-C: robustness of P1 (medium held-out, all lines, F vs path_distance) ===================")
    M = f"{ARMS}/medium_context/arms"
    rows = [json.loads(l) for l in open(f"{M}/h_F/generation_results.jsonl")]
    repo_of = {r["datapoint_idx"]: r["repo"] for r in rows}
    f = datapoint_means(arm_lines(M, "h_F"), "em", None)
    pd = datapoint_means(arm_lines(M, "h_path_distance"), "em", None)
    ids = sorted(set(f) & set(pd))
    diff = np.array([f[i] - pd[i] for i in ids])
    wins, losses, ties = int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())
    nontied = wins + losses
    # two-sided exact sign test
    k = min(wins, losses)
    p_sign = min(1.0, 2 * sum(math.comb(nontied, i) for i in range(k + 1)) / 2 ** nontied)
    print(f"(i) datapoints: {len(ids)} | F better: {wins} | F worse: {losses} | tied: {ties} | share of non-tied won by F: "
          f"{wins / nontied:.1%} | sign test p = {p_sign:.2g}")
    print(f"    mean diff {diff.mean():+.4f}; median {np.median(diff):+.4f}")
    repos = sorted({repo_of[i] for i in ids})
    by_repo = {r: [d for i, d in zip(ids, diff) if repo_of[i] == r] for r in repos}
    print(f"(ii) distinct repositories: {len(repos)} (datapoints per repo: max {max(len(v) for v in by_repo.values())})")
    repo_means = np.array([np.mean(by_repo[r]) for r in repos])
    rng = np.random.default_rng(0)
    # cluster bootstrap: resample repositories, keep all their datapoints, mean over datapoints
    boots = []
    for _ in range(10000):
        pick = rng.integers(0, len(repos), len(repos))
        vals = np.concatenate([by_repo[repos[j]] for j in pick])
        boots.append(vals.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"    repository-clustered bootstrap 95% CI of the mean difference: [{lo:+.4f}, {hi:+.4f}] (point {diff.mean():+.4f})")
    loo = [np.concatenate([by_repo[r2] for r2 in repos if r2 != r]).mean() for r in repos]
    print(f"(iii) leave-one-repository-out estimates: min {min(loo):+.4f}, max {max(loo):+.4f}")
    order = np.argsort(-diff)
    trimmed = np.delete(diff, order[:10]).mean()
    print(f"    removing the 10 datapoints where F gains most: mean difference {trimmed:+.4f}; "
          f"removing the 10 where it loses most: {np.delete(diff, order[-10:]).mean():+.4f}")

# ----------------------------------------------------------------------------------------------- V-D
if "D" in which:
    print("\n=================== V-D: where does the advantage come from? (medium held-out, 1500 random lines) ===================")
    M = f"{ARMS}/medium_context/arms"

    def em_rows(arm):
        out = {}
        for r in map(json.loads, open(f"{M}/{arm}/generation_results.jsonl")):
            if not r["zero_context"]:
                out[(r["datapoint_idx"], r["line_class"], r["line_num"])] = float(
                    r["prediction_line"].strip() == r["ground_truth"].strip())
        return out

    em_f, em_pd = em_rows("h_F"), em_rows("h_path_distance")
    rng = random.Random(7)
    keys = sorted(em_f)
    sample = rng.sample(keys, 1500)
    by_dp = {}
    for key in sample:
        by_dp.setdefault(key[0], []).append(key)
    composer = PathDistanceComposer(lang_sep_symbol="", meta_info_sep_symbol="METASEP\n", extension="")
    strata = {}
    for p, ks in by_dp.items():
        dp = medium[p]
        builder, path = builder_for(dp)
        static_context = composer.context_composer(dp)
        for (_, cls, ln) in ks:
            g = dp.get_line(ln).strip()
            prefix = dp.get_prefix(ln)
            r = builder.context_for(prefix, BUDGET, **F)
            f_copy = len(g) >= 15 and g in {x.strip() for x in r.context.split("\n")}
            prompt = "\n".join([static_context, prefix])
            seen = visible_prompt(prompt, encode, decode, BUDGET)
            ctx = seen[: len(seen) - len(prefix) - 1] if seen.endswith("\n" + prefix) else seen
            pd_copy = len(g) >= 15 and g in {x.strip() for x in ctx.split("\n")}
            strata.setdefault((f_copy, pd_copy), []).append(em_f[(p, cls, ln)] - em_pd[(p, cls, ln)])
            strata.setdefault(("em", f_copy, pd_copy), []).append((em_f[(p, cls, ln)], em_pd[(p, cls, ln)]))
    total = sum(sum(v) for k, v in strata.items() if len(k) == 2)
    n = sum(len(v) for k, v in strata.items() if len(k) == 2)
    print(f"{'ground truth visible in context?':36s}{'lines':>7s}{'EM F':>8s}{'EM pd':>8s}{'sum(F-pd)':>11s}{'share of advantage':>20s}")
    for (fc, pc) in [(True, True), (True, False), (False, True), (False, False)]:
        v = strata.get((fc, pc), [])
        e = strata.get(("em", fc, pc), [])
        if not v:
            continue
        label = f"F: {'yes' if fc else 'no '} | path_distance: {'yes' if pc else 'no '}"
        print(f"{label:36s}{len(v):7d}{np.mean([a for a, _ in e]):8.3f}{np.mean([b for _, b in e]):8.3f}{sum(v):+11.0f}"
              f"{(sum(v) / total if total else float('nan')):20.1%}")
    f_copy_lines = sum(len(strata.get((True, pc), [])) for pc in (True, False))
    f_copy_adv = sum(sum(strata.get((True, pc), [])) for pc in (True, False))
    print(f"lines (of {n}) where F's retrieved context contains the ground-truth line verbatim: {f_copy_lines} "
          f"({f_copy_lines / n:.1%}); they carry {f_copy_adv / total:.1%} of F's total advantage (sum {total:+.0f} lines)")
