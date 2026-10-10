"""Run the original CodeT5 CMG baseline and save reproducible results.

Generation, preprocessing, and scoring come from the existing CMG implementation.
Completed predictions can be evaluated later without generating them again.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parent
SEEDS = [2687987020, 42, 123]
PACKAGES = (
    "torch", "transformers", "datasets", "evaluate", "torchmetrics",
    "bert-score", "sacrebleu", "rouge-score", "numpy", "huggingface-hub",
)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_predictions(path, dataset):
    """Reject incomplete, reordered, duplicated, or mismatched result files."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != len(dataset):
        raise ValueError(
            f"{path} has {len(rows)} predictions; expected {len(dataset)}. "
            "Use --restart to regenerate an interrupted run from its seed."
        )
    for index, (prediction, source) in enumerate(zip(rows, dataset)):
        if any(prediction.get(key) != source[field] for key, field in (
            ("hash", "hash"), ("repo", "repo"), ("reference", "message"),
        )) or not isinstance(prediction.get("prediction"), str):
            raise ValueError(f"Invalid or mismatched prediction at row {index} in {path}")
    return rows


def load_config(device="cuda"):
    from omegaconf import OmegaConf

    settings = OmegaConf.create({"limit": None, "expected_dataset_size": 163, "cpu_threads": 2})
    baseline = OmegaConf.load(ROOT / "configs/examples/cmg_codet5.yaml")
    baseline.backbone.device = device
    baseline.backbone.model_kwargs.device_map = None
    baseline.logger.use_wandb = False
    return settings, baseline


def load_data(cfg, settings):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(cfg.data_src.hub_name).sha
    if list(cfg.data_src.configs) != ["default"] or cfg.data_src.split != "test":
        raise ValueError("Expected the default/test CMG dataset, containing commit diffs.")
    dataset = load_dataset(cfg.data_src.hub_name, "default", split="test", revision=revision)
    if len(dataset) != settings.expected_dataset_size:
        raise ValueError(f"Dataset has {len(dataset)} rows; expected {settings.expected_dataset_size}.")
    if settings.limit is not None:
        if settings.limit > len(dataset):
            raise ValueError("limit exceeds the dataset size")
        dataset = dataset.select(range(settings.limit))
    return dataset, revision


def run_seed(settings, cfg, dataset, revision, output_root, seed, args):
    import torch
    from omegaconf import OmegaConf
    from tqdm import tqdm

    from run_baseline import compute_metrics, init_baseline

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.backbone.seed = seed
    run_dir = output_root / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    config_path = run_dir / "config.yaml"
    manifest_path = run_dir / "run_metadata.json"
    config_text = OmegaConf.to_yaml(cfg, resolve=True)
    identity = {
        "config": config_text,
        "dataset_revision": revision,
        "dataset_fingerprint": dataset._fingerprint,
        "num_examples": len(dataset),
        "cpu_threads": int(settings.cpu_threads),
        "packages": {name: version(name) for name in PACKAGES},
    }
    if predictions_path.exists() and not args.restart:
        if not manifest_path.exists() or json.loads(manifest_path.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError(f"Run settings changed for {run_dir}. Use a new --output-dir or --restart.")
        read_predictions(predictions_path, dataset)
        print(f"Reusing complete predictions: {predictions_path}", flush=True)
    else:
        if args.evaluate_only:
            raise ValueError(f"No complete predictions available at {predictions_path}")
        # Discard stale scores when explicitly regenerating, even if generation fails.
        for name in ("metrics.json", "summary.json"):
            (run_dir / name).unlink(missing_ok=True)
        config_path.write_text(config_text, encoding="utf-8")
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        save_json(manifest_path, {
            "identity": identity,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "source_commit": commit,
            "source_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [ROOT / "baseline_replication.py", ROOT / "run_baseline.py", *sorted((ROOT / "src").rglob("*.py"))]
            },
        })
        baseline = init_baseline(cfg)
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata["model_revision"] = getattr(baseline._backbone._model.config, "_commit_hash", None)
        metadata["tokenizer_revision"] = baseline._backbone._tokenizer.init_kwargs.get("_commit_hash")
        if cfg.backbone.device == "cuda":
            metadata["cuda"] = {"runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
        save_json(manifest_path, metadata)
        # Match the original runner's sequential sampling, without batching or changing output length.
        with predictions_path.open("w", encoding="utf-8") as writer:
            for row in tqdm(dataset, desc=f"Generating (seed {seed})"):
                result = {"reference": row["message"], "hash": row["hash"], "repo": row["repo"]}
                result.update(baseline.generate_msg(commit_mods=row["mods"]))
                if not isinstance(result.get("prediction"), str):
                    raise ValueError(f"Invalid prediction for {row['hash']}")
                writer.write(json.dumps(result, ensure_ascii=False) + "\n")
                writer.flush()
        del baseline
        gc.collect()  # Release CodeT5 before loading BERTScore's larger scoring model.
        if cfg.backbone.device == "cuda":
            torch.cuda.empty_cache()

    predictions = read_predictions(predictions_path, dataset)
    full = len(dataset) == settings.expected_dataset_size and settings.limit is None
    summary = {
        "model": cfg.backbone.model_name,
        "dataset": cfg.data_src.hub_name,
        "dataset_config": "default",
        "split": "test",
        "seed": seed,
        "device": cfg.backbone.device,
        "num_predictions": len(predictions),
        "scope": "full test set" if full else "partial run; not a full benchmark score",
        "full_test_set": full,
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "metrics_status": "not computed",
        "comparison_note": f"Replication of the released configuration on {cfg.backbone.device}; exact paper-score reproduction is not verified.",
    }
    save_json(run_dir / "summary.json", summary)
    if not args.generate_only:
        from transformers import set_seed

        # ROUGE uses bootstrap sampling: make generation-time and deferred scoring agree.
        set_seed(seed)
        # The original metrics include B-Norm, BLEU, chrF, ROUGE, and both BERTScores.
        metrics = {key: float(value) for key, value in compute_metrics(str(predictions_path)).items()}
        save_json(run_dir / "metrics.json", metrics)
        summary.update(metrics_status="complete", metrics=metrics)
        save_json(run_dir / "summary.json", summary)
    print(f"Saved run: {run_dir}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="Default: cuda (NVIDIA GPU with CUDA-enabled PyTorch); use --device cpu for CPU")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="Default: 2687987020 42 123")
    parser.add_argument("--output-dir", type=Path, help="Default: outputs/codet5_full for CPU; outputs/codet5_full_cuda for CUDA")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate-only", action="store_true", help="Save predictions; defer expensive BERTScore evaluation")
    mode.add_argument("--evaluate-only", action="store_true", help="Evaluate complete saved predictions without loading CodeT5")
    parser.add_argument("--restart", action="store_true", help="Regenerate predictions from the beginning, overwriting this run")
    args = parser.parse_args()
    if args.restart and args.evaluate_only:
        parser.error("--restart and --evaluate-only cannot be combined")

    # Keep model, dataset, metric-script, and plotting caches in the project.
    for key, relative in (
        ("HF_HOME", "huggingface"), ("XDG_CACHE_HOME", "xdg"), ("MPLCONFIGDIR", "matplotlib"),
    ):
        os.environ.setdefault(key, str(ROOT / "cache" / relative))

    import torch

    cuda_available = torch.cuda.is_available()
    if args.device == "cuda" and not cuda_available:
        parser.error("CUDA is unavailable. Use an NVIDIA GPU with CUDA-enabled PyTorch, or select --device cpu.")
    print(f"Using device: {args.device}", flush=True)
    settings, cfg = load_config(args.device)
    seeds = args.seeds
    if len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 2**32 for seed in seeds):
        parser.error("seeds must be distinct integers between 0 and 2**32 - 1")
    torch.set_num_threads(int(settings.cpu_threads))
    output_root = (args.output_dir or ROOT / "outputs" / ("codet5_full_cuda" if args.device == "cuda" else "codet5_full")).resolve()
    print("Loading the CMG default/test dataset...", flush=True)
    dataset, revision = load_data(cfg, settings)
    # Clearing an aggregate ensures an interrupted rerun cannot leave an old success report.
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "aggregate.json").unlink(missing_ok=True)
    summaries = [run_seed(settings, cfg, dataset, revision, output_root, seed, args) for seed in seeds]
    if len(summaries) > 1 and all(row["metrics_status"] == "complete" for row in summaries):
        metric_names = summaries[0]["metrics"].keys()
        save_json(output_root / "aggregate.json", {
            "device": cfg.backbone.device,
            "seeds": seeds,
            "num_examples_per_seed": len(dataset),
            "full_test_set": all(row["full_test_set"] for row in summaries),
            "scope": summaries[0]["scope"],
            "mean": {key: mean(row["metrics"][key] for row in summaries) for key in metric_names},
            "population_stddev": {key: pstdev(row["metrics"][key] for row in summaries) for key in metric_names},
            "seed_note": "Only 2687987020 is given in the original example; additional seeds are user-selected.",
        })
        print(f"Average scores saved: {output_root / 'aggregate.json'}", flush=True)


if __name__ == "__main__":
    main()
