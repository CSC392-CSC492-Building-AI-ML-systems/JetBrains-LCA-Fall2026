"""Does every registered composer run on a real (development) datapoint? CPU only."""
import importlib
import os
import time

from hydra import compose, initialize_config_dir

from composers.composer_registry import COMPOSERS
from eval.line_generators import dev_indices
from eval.preprocessors import load_datapoints

with initialize_config_dir(config_dir=os.path.join(os.environ["PROJECT"], "eval", "config"), version_base=None):
    cfg = compose(config_name="compare_arms", overrides=["dataset=medium"])
data = load_datapoints(cfg.dataset)
dp = data[dev_indices(len(data), 28, 28)[3]]  # a development datapoint
args = dict(cfg.composers_config)
print("datapoint:", dp.repo_name, "| files in snapshot:", len(dp.get_context()))
for name, spec in COMPOSERS.items():
    if spec is None:
        print(f"{name:28s} (no context)")
        continue
    t0 = time.time()
    try:
        composer = getattr(importlib.import_module(spec["module"]), spec["name"])(**args)
        text = composer.context_composer(dp)
        again = composer.context_composer(dp)
        print(f"{name:28s} ok  {len(text):9d} chars  {time.time() - t0:5.2f}s  repeatable-in-process: {text == again}")
    except Exception as error:  # report, do not stop
        print(f"{name:28s} FAILED: {type(error).__name__}: {str(error)[:120]}")
