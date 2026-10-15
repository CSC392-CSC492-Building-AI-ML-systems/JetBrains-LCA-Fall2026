"""Does the 'no room for chunks' edge case (prefix + header/tail above the budget) affect the held-out result? (CPU)"""
import json
import os

import numpy as np
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer

from eval.analyze_arms import paired_bootstrap
from eval.line_generators import heldout_indices
from eval.preprocessors import load_datapoints

BUDGET = 7900
HOME = os.path.expanduser("~")
tok = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b")
count = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])

for dataset, dev_size, dev_extra, ds_dir in (("medium", 28, 28, "medium_context"), ("small", 20, 28, "small_context")):
    with initialize_config_dir(config_dir=os.path.join(os.environ["PROJECT"], "eval", "config"), version_base=None):
        cfg = compose(config_name="compare_arms", overrides=[f"dataset={dataset}"])
    data = load_datapoints(cfg.dataset)
    held = [data[i] for i in heldout_indices(len(data), dev_size, dev_extra)]
    noroom = set()
    for p, dp in enumerate(held):
        path = next(iter(dp.get_completion()))
        fixed = count(f"{dp.repo_name}METASEP\n" + path + "METASEP\n" + "\n")
        for cls, nums in dp.completion_lines.items():
            for ln in nums:
                prefix = dp.get_prefix(ln)
                if len(prefix) > BUDGET * 0.9 and count(prefix) + fixed > BUDGET:
                    noroom.add((p, cls, ln))
    base = f"{HOME}/lca_artifacts/python/starcoderbase-1b/{ds_dir}/arms"

    def em(arm):
        out = {}
        for r in map(json.loads, open(f"{base}/{arm}/generation_results.jsonl")):
            if bool(r["zero_context"]) == (arm == "h_zero"):  # the zero arm only has zero-context rows
                out[(r["datapoint_idx"], r["line_class"], r["line_num"])] = float(
                    r["prediction_line"].strip() == r["ground_truth"].strip())
        return out

    arms = {a: em(a) for a in ("h_zero", "h_path_distance", "h_F")}
    keys = sorted(arms["h_F"])
    in_room = [k for k in keys if k not in noroom]

    def dp_means(arm, selected):
        per = {}
        for k in selected:
            per.setdefault(k[0], []).append(arms[arm][k])
        return {p: float(np.mean(v)) for p, v in per.items()}

    print(f"== {dataset} held-out: {len(noroom)} no-room lines of {len(keys)} ==")
    nr_keys = [k for k in keys if k in noroom]
    if nr_keys:
        print("   EM on the no-room lines: " + ", ".join(f"{a} {np.mean([arms[a][k] for k in nr_keys]):.3f}" for a in arms))
        same = sum(arms["h_F"][k] == arms["h_path_distance"][k] for k in nr_keys)
        print(f"   F and path_distance agree in outcome on {same} of {len(nr_keys)} of these lines")
    for label, sel in (("all lines", keys), ("without the no-room lines", in_room)):
        res = paired_bootstrap(dp_means("h_path_distance", sel), dp_means("h_F", sel))
        d, lo, hi, n = res
        print(f"   F - path_distance, {label}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] over {n} datapoints")
