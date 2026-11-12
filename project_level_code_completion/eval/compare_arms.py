"""Run one arm of the context comparison (zero context, a static composer, or RAG) under identical settings.

Example:
    poetry run python -m eval.compare_arms arm=rag limit_datapoints=20
    poetry run python -m eval.compare_arms arm=path_distance limit_datapoints=20
"""
import importlib
import json
import os
import shutil
import subprocess
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from agent.llm import LocalChatModel
from agent.provider import make_agent_provider_factory
from agent.tools import Limits
from composers.composer_registry import COMPOSERS
from eval.line_generators import (GeneratorConfig, dev_indices, evaluate_generation, heldout_indices, shard_indices,
                                  subset_indices)
from eval.proxy_metrics import targeted_lines
from eval.preprocess import preprocess
from eval.preprocessors import load_datapoints
from eval.run_metrics import cost_block, crop_to_last_tokens
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
    allowed = ['zero', 'rag', 'agent'] + [name for name in COMPOSERS if name != 'none']
    if cfg.arm not in allowed:
        raise ValueError(f"Unknown arm {cfg.arm!r}. Available: {allowed}")
    if cfg.static_mode not in ('provider', 'preprocess'):
        raise ValueError(f"static_mode must be provider or preprocess, got {cfg.static_mode!r}")
    if cfg.lines not in ('all', 'targeted'):
        raise ValueError(f"lines must be all or targeted, got {cfg.lines!r}")
    if cfg.split not in ('all', 'dev', 'heldout'):
        raise ValueError(f"split must be all, dev or heldout, got {cfg.split!r}")
    if cfg.split != 'all' and cfg.limit_datapoints is not None:
        raise ValueError("limit_datapoints cannot be combined with split=dev/heldout (use dev_size)")
    shard_index, shard_count = cfg.get('shard_index'), cfg.get('shard_count')  # absent in older configs
    if (shard_index is None) != (shard_count is None):
        raise ValueError("shard_index and shard_count go together")
    if shard_count is not None:
        if shard_count < 1 or not 0 <= shard_index < shard_count:
            raise ValueError(f"need 0 <= shard_index < shard_count, got {shard_index!r} of {shard_count!r}")
        if cfg.split != 'all' or cfg.limit_datapoints is not None:
            raise ValueError("shards split the whole dataset: use split=all and no limit_datapoints")
    if cfg.get('resume'):  # absent in older configs
        if cfg.overwrite:
            raise ValueError("resume=true keeps the existing results, overwrite=true deletes them: pick one")
        if cfg.arm == 'zero' or cfg.run_zero:
            raise ValueError("resume only continues the context pass (the zero-context pass is not saved apart)")
    cap = cfg.get('context_cap')  # absent in older configs
    if cap is not None:
        if cap < 1:
            raise ValueError(f"context_cap must be a positive number of tokens, got {cap!r}")
        if cfg.arm not in ('zero', 'rag', 'agent') and cfg.static_mode != 'provider':
            raise ValueError("context_cap needs static_mode=provider for static composers")
    if cfg.arm == 'agent' and (cfg.rag.retriever != 'bm25' or cfg.rag.chunking != 'structure'):
        raise ValueError("arm=agent edits the context of RAG with the BM25 retriever and structure chunking")


def datapoint_selection(cfg: DictConfig) -> dict:
    """dev = `dev_size` evenly spaced datapoints plus `dev_extra` more; heldout = all the others;
    all = everything (or `limit_datapoints`); with shard_index/shard_count, every shard_count-th datapoint from
    shard_index (saved `datapoint_idx` is then the position inside the shard; `merge_shards.py` restores the original)."""
    if cfg.get('shard_count') is not None:
        return {'select': partial(shard_indices, shard_index=cfg.shard_index, shard_count=cfg.shard_count)}
    if cfg.split == 'dev':
        return {'select': partial(dev_indices, dev_size=cfg.dev_size, dev_extra=cfg.dev_extra)}
    if cfg.split == 'heldout':
        return {'select': partial(heldout_indices, dev_size=cfg.dev_size, dev_extra=cfg.dev_extra)}
    return {'limit': cfg.limit_datapoints}


def static_provider_factory(kind: str, composer_args: dict, cap_tokens=None, tokenizer=None):
    """Provider factory for a static composer: the composed context string, a newline, then the prefix.

    This is the text the generator builds from a preprocessed datapoint (`"\\n".join([context, prefix])`), without
    tokenizing the whole dataset first. `evaluate_generation` calls the factory once per datapoint.
    """
    spec = COMPOSERS[kind]
    composer = getattr(importlib.import_module(spec['module']), spec['name'])(**composer_args)

    def factory(datapoint):
        context = composer.context_composer(datapoint)
        if cap_tokens:
            context = crop_to_last_tokens(context, cap_tokens, tokenizer)
        return lambda dp, line_num: ("\n".join([context, dp.get_prefix(line_num)]), dp.get_line(line_num), None)

    return factory


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
    context_cap = cfg.get('context_cap')
    arm_dir = arm_directory(cfg)
    results_path = arm_dir / 'generation_results.jsonl'
    resume = bool(cfg.get('resume'))
    if results_path.exists() and not resume:
        if not cfg.overwrite:
            raise FileExistsError(f"{results_path} exists; use a different arm_name or overwrite=true or resume=true")
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
                  resume=resume, **datapoint_selection(cfg))
    if cfg.arm in ('zero', 'rag', 'agent'):
        kwargs['input_data'] = load_datapoints(cfg.dataset)
    elif cfg.static_mode == 'provider':
        kwargs['input_data'] = load_datapoints(cfg.dataset)
        cap_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) if context_cap else None
        kwargs['context_provider_factory'] = static_provider_factory(
            cfg.arm, dict(cfg.composers_config), cap_tokens=context_cap, tokenizer=cap_tokenizer)
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
        rag_options = OmegaConf.to_container(cfg.rag)
        if context_cap is not None:
            rag_options['max_context_tokens'] = context_cap  # same context size N as the static composers
        kwargs['context_provider_factory'] = make_rag_provider_factory(
            lambda text: len(tokenizer(text, add_special_tokens=False)['input_ids']), budget,
            extension=cfg.composers_config.extension, lang_sep=cfg.composers_config.lang_sep_symbol,
            meta_sep=cfg.composers_config.meta_info_sep_symbol, **rag_options,
        )
    if cfg.arm == 'agent':
        # The planner starts from exactly the context the matching RAG arm would build (same options, same cap N).
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        rag_options = OmegaConf.to_container(cfg.rag)
        if context_cap is not None:
            rag_options['max_context_tokens'] = context_cap
        planner = LocalChatModel(cfg.agent.planner_model, max_new_tokens=cfg.agent.planner_max_new_tokens,
                                 dtype=cfg.agent.planner_dtype, device=cfg.params.eval_params['device'])
        search_embedder = None
        if cfg.agent.get('search_embedder'):  # dense ranking for the hybrid search tool (loaded on first use)
            from rag.dense import DenseEmbedder
            search_embedder = DenseEmbedder(cfg.agent.search_embedder, pooling=cfg.agent.get('search_pooling', 'cls'),
                                            cache_dir=cfg.agent.get('search_cache'))
        kwargs['context_provider_factory'] = make_agent_provider_factory(
            lambda text: len(tokenizer(text, add_special_tokens=False)['input_ids']), budget, planner,
            rag_options=rag_options, limits=Limits(**OmegaConf.to_container(cfg.agent.limits)),
            planner_prefix_tokens=cfg.agent.planner_prefix_tokens, max_listed_files=cfg.agent.max_listed_files,
            extension=cfg.composers_config.extension, lang_sep=cfg.composers_config.lang_sep_symbol,
            meta_sep=cfg.composers_config.meta_info_sep_symbol, search_embedder=search_embedder,
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    if cfg.lines == 'targeted':
        # Only lines whose needed names are defined in other files (log section 8.13); computed from the snapshot.
        snapshot = kwargs.get('input_data') or load_datapoints(cfg.dataset)
        kwargs['select'] = kwargs.get('select') or partial(subset_indices, limit=cfg.limit_datapoints)
        kwargs.pop('limit', None)
        kwargs['line_filter'] = lambda i: targeted_lines(snapshot[i])

    scores, score_lists, em_diff, line_counts = evaluate_generation(generator_config, **kwargs)

    wall_seconds = round(time.time() - started, 1)
    has_gpu = torch.cuda.is_available()
    cost = cost_block(str(results_path), wall_seconds,
                      torch.cuda.get_device_name(0) if has_gpu else None,
                      round(torch.cuda.max_memory_allocated() / 1e9, 2) if has_gpu else None)
    summary = {'arm': cfg.arm, 'arm_name': cfg.arm_name or cfg.arm, 'dataset': dataset_name(cfg),
               'model': inference_params['model'], 'token_budget': budget, 'context_cap': context_cap,
               'limit_datapoints': cfg.limit_datapoints,
               'split': cfg.split, 'dev_size': cfg.dev_size, 'dev_extra': cfg.dev_extra, 'lines': cfg.lines,
               'static_mode': cfg.static_mode,
               'shard_index': cfg.get('shard_index'), 'shard_count': cfg.get('shard_count'), 'resumed': resume,
               'code': git_state(), 'wall_seconds': wall_seconds, 'cost': cost,
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
