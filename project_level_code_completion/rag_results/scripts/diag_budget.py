"""Why do some F prompts exceed the budget? And how many held-out lines are affected? (CPU)"""
import os
import random

from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer

from eval.line_generators import heldout_indices
from eval.preprocessors import load_datapoints
from rag.builder import RagContextBuilder

BUDGET = 7900
F = dict(prefix_cap=7900, continuation_lines=20)
tok = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b")
count = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])


def heldout(dataset, dev_size, dev_extra):
    with initialize_config_dir(config_dir=os.path.join(os.environ["PROJECT"], "eval", "config"), version_base=None):
        cfg = compose(config_name="compare_arms", overrides=[f"dataset={dataset}"])
    data = load_datapoints(cfg.dataset)
    return [data[i] for i in heldout_indices(len(data), dev_size, dev_extra)]


# 1) the violating prompts of the verification sample (same sampling as verify_cpu.py, check B)
medium, small = heldout("medium", 28, 28), heldout("small", 20, 28)
rng = random.Random(123)
plan = [("medium", p) for p in rng.sample(range(len(medium)), 70)] + [("small", p) for p in rng.sample(range(len(small)), 30)]
print("violating prompts:")
for name, p in plan:
    dp = (medium if name == "medium" else small)[p]
    path = next(iter(dp.get_completion()))
    builder = RagContextBuilder(dp.get_context(), path, dp.repo_name, count, lang_sep="", meta_sep="METASEP\n")
    lines = [(cls, ln) for cls, nums in dp.completion_lines.items() for ln in nums]
    for cls, ln in rng.sample(lines, min(4, len(lines))):
        prefix = dp.get_prefix(ln)
        r = builder.context_for(prefix, BUDGET, **F)
        total = count(r.context + "\n" + r.prefix)
        if total > BUDGET:
            print(f"  {name} dp{p} line {ln}: total {total} = prefix {count(r.prefix)} + header/tail "
                  f"{count(r.context)}; chunks selected: {len(r.chunks)}")

# 2) how many held-out lines have a prefix so long that nothing can be added? (prefix + header/tail > budget)
for name, data in (("medium", medium), ("small", small)):
    n = over = 0
    for dp in data:
        path = next(iter(dp.get_completion()))
        fixed = count(f"{dp.repo_name}METASEP\n" + path + "METASEP\n" + "\n")
        for nums in dp.completion_lines.values():
            for ln in nums:
                n += 1
                prefix = dp.get_prefix(ln)
                if len(prefix) > BUDGET * 0.9:  # cheap pre-filter: a token is >= 1 char, so tokenize only long prefixes
                    over += count(prefix) + fixed > BUDGET
    print(f"{name} held-out: {over} of {n} lines ({over / n:.2%}) have prefix + header/tail above the budget")
