"""Run one arm of the context comparison (zero context, a static composer, or RAG) under identical settings.

Example:
    poetry run python -m eval.compare_arms arm=rag limit_datapoints=20
    poetry run python -m eval.compare_arms arm=path_distance limit_datapoints=20
"""
import json
import os
import shutil
import subprocess
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from composers.composer_registry import COMPOSERS
from eval.line_generators import GeneratorConfig, dev_indices, evaluate_generation, heldout_indices
from eval.preprocess import preprocess
from eval.preprocessors import load_datapoints
from rag.provider import make_rag_provider_factory

# Same reserve the main pipeline leaves for the generated line.
GENERATION_RESERVE = 100


def dataset_name(cfg: DictConfig) -> str:
    if isinstance(cfg.dataset, str):
        return cfg.dataset.split('/')[-1].replace('-', '_')
    return cfg.dataset['name']


def arm_directory(cfg: DictConfig) -> Path:
    model = cfg.params.inference_params['model']
    return Path(cfg.artifacts_dir, cfg.language, model, dataset_name(cfg), 'arms', cfg.arm_name or cfg.arm)


def validate(cfg: DictConfig) -> None:
    allowed = ['zero', 'rag'] + [name for name in COMPOSERS if name != 'none']
    if cfg.arm not in allowed:
        raise ValueError(f"Unknown arm {cfg.arm!r}. Available: {allowed}")
    if cfg.split not in ('all', 'dev', 'heldout'):
        raise ValueError(f"split must be all, dev or heldout, got {cfg.split!r}")
    if cfg.split != 'all' and cfg.limit_datapoints is not None:
        raise ValueError("limit_datapoints cannot be combined with split=dev/heldout (use dev_size)")


def datapoint_selection(cfg: DictConfig) -> dict:
    """dev = `dev_size` evenly spaced datapoints plus `dev_extra` more; heldout = all the others;
    all = everything (or `limit_datapoints`)."""
    if cfg.split == 'dev':
        return {'select': partial(dev_indices, dev_size=cfg.dev_size, dev_extra=cfg.dev_extra)}
    if cfg.split == 'heldout':
        return {'select': partial(heldout_indices, dev_size=cfg.dev_size, dev_extra=cfg.dev_extra)}
    return {'limit': cfg.limit_datapoints}


def git_state() -> dict:
    """Commit and dirty flag of the code that produced a result."""
    def git(*args):
        project_dir = str(Path(__file__).resolve().parent.parent)
        return subprocess.run(['git', '-C', project_dir, *args], capture_output=True, text=True).stdout.strip()

    try:
        return {'commit': git('rev-parse', '--short', 'HEAD'), 'dirty': bool(git('status', '--porcelain', '--', '.'))}
    except OSError:
        return {'commit': None, 'dirty': None}


def run_arm(cfg: DictConfig):
    validate(cfg)
    arm_dir = arm_directory(cfg)
    results_path = arm_dir / 'generation_results.jsonl'
    if results_path.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"{results_path} exists; use a different arm_name or overwrite=true")
        shutil.rmtree(arm_dir)
    arm_dir.mkdir(parents=True, exist_ok=True)

    inference_params = cfg.params.inference_params
    tokenizer_path = cfg.params.preprocess_params['tokenizer']
    budget = inference_params['seq_max_len'] - GENERATION_RESERVE
    generator_config = GeneratorConfig(
        input_data_path='', seq_max_len=budget, context_max=budget, model=inference_params['model'],
        device=cfg.params.eval_params['device'], best_perplexity=0.0, tokenizer_path=tokenizer_path,
        composer=cfg.arm, seed=cfg.seed, results_path=str(results_path),
    )

    kwargs = dict(run_zero=cfg.arm == 'zero' or cfg.run_zero, run_context=cfg.arm != 'zero',
                  **datapoint_selection(cfg))
    if cfg.arm in ('zero', 'rag'):
        kwargs['input_data'] = load_datapoints(cfg.dataset)
    else:
        # Static composer: preprocess the whole dataset with it, exactly as the main pipeline does.
        in_dir = arm_dir / 'in'
        in_dir.mkdir(parents=True, exist_ok=True)
        preprocess_args = SimpleNamespace(
            model=cfg.params.preprocess_params['model'], dataset=cfg.dataset, tokenizer=tokenizer_path,
            composers=cfg.arm, out_dir=str(in_dir), context_len_char=5 * inference_params['seq_max_len'],
        )
        generator_config.input_data_path = preprocess(preprocess_args, cfg.composers_config)
    if cfg.arm == 'rag':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        kwargs['context_provider_factory'] = make_rag_provider_factory(
            lambda text: len(tokenizer(text, add_special_tokens=False)['input_ids']), budget,
            prefix_cap=cfg.rag.prefix_cap, query_lines=cfg.rag.query_lines, max_chunks=cfg.rag.max_chunks,
            include_non_py=cfg.rag.include_non_py, extension=cfg.composers_config.extension,
            lang_sep=cfg.composers_config.lang_sep_symbol, meta_sep=cfg.composers_config.meta_info_sep_symbol,
        )

    started = time.time()
    scores, score_lists, em_diff, line_counts = evaluate_generation(generator_config, **kwargs)

    summary = {'arm': cfg.arm, 'arm_name': cfg.arm_name or cfg.arm, 'dataset': dataset_name(cfg),
               'model': inference_params['model'], 'token_budget': budget, 'limit_datapoints': cfg.limit_datapoints,
               'split': cfg.split, 'dev_size': cfg.dev_size, 'dev_extra': cfg.dev_extra,
               'code': git_state(), 'wall_seconds': round(time.time() - started, 1),
               'scores': scores, 'em_difference_vs_zero': em_diff, 'line_counts': line_counts}
    (arm_dir / 'summary.json').write_text(json.dumps(summary, indent=4))
    (arm_dir / 'per_datapoint_scores.json').write_text(json.dumps(score_lists))
    (arm_dir / 'config.json').write_text(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=4))
    print(f">>Results are in {os.fspath(arm_dir)}")
    return summary


@hydra.main(config_path="config", config_name="compare_arms", version_base=None)
def main(cfg: DictConfig) -> None:
    print(json.dumps(run_arm(cfg), indent=4))


if __name__ == '__main__':
    main()
