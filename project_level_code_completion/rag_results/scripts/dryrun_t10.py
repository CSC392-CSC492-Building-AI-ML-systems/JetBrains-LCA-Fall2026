"""Dry run of the repository's own pipeline setup for the T10 command (no GPU, no data download of models)."""
import os

from hydra import compose, initialize_config_dir

from composers.composer_registry import COMPOSERS
from eval.eval_pipeline import EvalPipeline

with initialize_config_dir(config_dir=os.path.join(os.environ["PROJECT"], "eval", "config"), version_base=None):
    cfg = compose(config_name="config", overrides=["dataset=medium", "params=codellama7b", "do_generation=False",
                                                   "artifacts_dir=/tmp/t10_dryrun"])
p = EvalPipeline(cfg)
print("model:", p.inference_args.model, "| seq_max_len:", p.inference_args.seq_max_len, "| tokenizer:", p.preprocess_args.tokenizer)
print("dataset:", p.dataset_name, "| composers to run:", p.composers)
print("composers in registry:", list(COMPOSERS))
print("out dir:", p.out_dir)
first, final = 256, p.inference_args.seq_max_len * 3 // 4
step = (final / first) ** 0.5
sizes, c = [], first
while c <= p.inference_args.seq_max_len:
    sizes.append(c)
    c = int(c * step) + 1
print("context sizes the pipeline will evaluate per composer:", sizes)
