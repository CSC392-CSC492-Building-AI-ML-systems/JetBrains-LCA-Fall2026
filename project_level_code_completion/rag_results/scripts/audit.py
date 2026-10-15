"""Pre-held-out audit, development datapoints only: (1) old copy of the completion file in the snapshot? (2) is the
tree-sitter chunker active?"""
import os
from collections import Counter

from hydra import compose, initialize_config_dir

from eval.line_generators import dev_indices
from eval.preprocessors import load_datapoints
from rag.builder import RagContextBuilder

for dataset, dev_size, dev_extra in (("small", 20, 28), ("medium", 28, 28)):
    with initialize_config_dir(config_dir=os.path.join(os.environ["PROJECT"], "eval", "config"), version_base=None):
        cfg = compose(config_name="compare_arms", overrides=[f"dataset={dataset}"])
    data = load_datapoints(cfg.dataset)
    dev = [data[i] for i in dev_indices(len(data), dev_size, dev_extra)]  # development datapoints only
    present = 0
    n_lines = n_in_old = 0
    n_t_lines = 0
    for dp in dev:
        path = next(iter(dp.get_completion()))
        snapshot = dp.get_context()
        if path in snapshot:
            present += 1
            old = {line.strip() for line in snapshot[path].split("\n") if line.strip()}
            for nums in dp.completion_lines.values():
                for n in nums:
                    gt = dp.get_line(n).strip()
                    if gt:
                        n_lines += 1
                        n_in_old += gt in old
    print(f"{dataset}: completion file present in the snapshot for {present} of {len(dev)} dev datapoints; "
          f"of {n_lines} target lines in those, {n_in_old} ({n_in_old / max(1, n_lines):.0%}) also occur verbatim in "
          f"the old copy")

# (2) is tree-sitter active? Build one real index and look at the chunk kinds.
dp = dev[0]
b = RagContextBuilder(dp.get_context(), next(iter(dp.get_completion())), dp.repo_name, lambda t: len(t) // 4)
print("chunk kinds with the grammar in the working directory:", dict(Counter(c.kind for c in b.chunks)))
print("symbols recorded:", sum(1 for c in b.chunks if c.symbol), "of", len(b.chunks))
