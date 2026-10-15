"""How many dev lines have cross-file names under the strict (not in prefix) vs loose (not defined in the file) rule."""
import os
from collections import Counter

from hydra import compose, initialize_config_dir

from eval.line_generators import dev_indices
from eval.preprocessors import load_datapoints
from eval.proxy_metrics import definitions, identifiers, repo_definitions

with initialize_config_dir(config_dir=os.path.join(os.environ["PROJECT"], "eval", "config"), version_base=None):
    cfg = compose(config_name="compare_arms")
data = load_datapoints(cfg.dataset)
dev = [data[i] for i in dev_indices(len(data), cfg.dev_size, cfg.dev_extra)]  # development datapoints only

total, strict, loose = Counter(), Counter(), Counter()
examples = []
for dp in dev:
    path = next(iter(dp.get_completion()))
    defined = repo_definitions(dp.get_context(), exclude_path=path)
    for cls, nums in dp.completion_lines.items():
        for n in nums:
            line, prefix = dp.get_line(n), dp.get_prefix(n)
            ids = identifiers(line) & defined
            total[cls] += 1
            s = ids - identifiers(prefix)
            l = ids - definitions(prefix)
            strict[cls] += bool(s)
            loose[cls] += bool(l)
            if cls == "inproject" and l and len(examples) < 4:
                examples.append((dp.repo_name, line.strip()[:90], sorted(l)[:4], bool(s)))
print(f"{'category':16s}{'lines':>7s}{'strict X':>10s}{'loose X':>9s}")
for cls in sorted(total):
    print(f"{cls:16s}{total[cls]:7d}{strict[cls]:10d}{loose[cls]:9d}")
print(f"{'ALL':16s}{sum(total.values()):7d}{sum(strict.values()):10d}{sum(loose.values()):9d}")
print("\nexamples of inproject lines under the loose rule:")
for e in examples:
    print("  ", e)
