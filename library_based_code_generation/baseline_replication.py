"""Reproducible runner for the released Library-based Code Generation baselines.

The model classes, prompts, code extraction, and metrics are the benchmark's
existing implementations. Predictions are stored in per-model subdirectories
and can be scored again without making API calls.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

from src.evaluation.evaluate import extract_code
from src.metrics.chrf import ChrF
from src.metrics.overlap import Overlap
from src.models.openai_model import OpenAIModel
from src.models.together_model import TogetherModel


ROOT = Path(__file__).resolve().parent
DATASET = "JetBrains-Research/lca-library-based-code-generation"
MODELS = (
    "gpt-3.5-turbo-0125",
    "gpt-4-0125-preview",
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-70b-Instruct-hf",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
)
SOURCE_FILES = (
    ROOT / "baseline_replication.py",
    ROOT / "src/evaluation/evaluate.py",
    ROOT / "src/models/example_generation_model.py",
    ROOT / "src/models/openai_model.py",
    ROOT / "src/models/together_model.py",
    ROOT / "src/metrics/chrf.py",
    ROOT / "src/metrics/overlap.py",
)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def package_versions():
    result = {}
    for package in ("datasets", "huggingface-hub", "numpy", "openai", "sacrebleu", "together", "transformers"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def load_data(limit):
    revision = HfApi().dataset_info(DATASET).sha
    dataset = load_dataset(DATASET, split="test", revision=revision)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset, revision


def read_existing(predictions_dir, dataset):
    predictions = []
    for index, _ in enumerate(dataset):
        path = predictions_dir / f"{index}.py"
        if not path.is_file():
            raise ValueError(f"Missing prediction {path}; rerun without --evaluate-only to finish generation.")
        prediction = path.read_text(encoding="utf-8")
        predictions.append(prediction)
    return predictions


def run_model(model_name, dataset, revision, output_root, args):
    display_name = "bm25/" + model_name
    run_dir = output_root / display_name
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "run_metadata.json"
    summary_path = run_dir / "summary.json"
    identity = {
        "dataset": DATASET,
        "dataset_revision": revision,
        "dataset_fingerprint": dataset._fingerprint,
        "split": "test",
        "num_examples": len(dataset),
        "full_test_set": args.limit is None,
        "model": model_name,
        "retrieval": "bm25 top-20",
        "temperature": 0.0,
        "max_tokens": 2048,
        "packages": package_versions(),
    }
    if args.evaluate_only:
        if not metadata_path.is_file() or json.loads(metadata_path.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError(f"Saved run settings do not match {run_dir}; use the original dataset/settings.")
        raw_predictions = read_existing(predictions_dir, dataset)
    else:
        if not args.restart and metadata_path.exists() and json.loads(metadata_path.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError(f"Run settings changed for {run_dir}; choose another --output-dir or use --restart.")
        if args.restart:
            for path in predictions_dir.glob("*.py"):
                path.unlink()
        save_json(metadata_path, {
            "identity": identity,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "source_commit": subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
            ).stdout.strip() or None,
            "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in SOURCE_FILES},
        })
        is_openai = model_name.startswith("gpt-")
        required_key = "OPENAI_API_KEY" if is_openai else "TOGETHER_API_KEY"
        if not os.environ.get(required_key):
            raise RuntimeError(f"{required_key} must be set in the environment before generation.")
        model = OpenAIModel(model_name, use_bm25=True) if is_openai else TogetherModel(model_name, use_bm25=True)
        raw_predictions = []
        for index, sample in enumerate(tqdm(dataset, desc=f"Generate {display_name}")):
            path = predictions_dir / f"{index}.py"
            if path.exists():
                generated = path.read_text(encoding="utf-8")
            else:
                generated = model.generate(sample["instruction"], sample["project_defined_elements"])
                path.write_text(generated, encoding="utf-8")
            raw_predictions.append(generated)

    metrics = {"ChrF": [], "API_recall": []}
    chrf, api_recall = ChrF(), Overlap()
    row_scores = []
    for index, (raw, sample) in enumerate(zip(raw_predictions, dataset)):
        code = extract_code(raw)
        scores = {
            chrf.name(): float(chrf.score(code, sample["clean_reference"], sample["unique_apis"])),
            api_recall.name(): float(api_recall.score(code, sample["clean_reference"], sample["unique_apis"])),
        }
        for name, score in scores.items():
            metrics[name].append(score)
        row_scores.append({"index": index, **scores})
    summary = {
        **identity,
        "scope": "full test set" if args.limit is None else "partial run; not a full benchmark score",
        "num_predictions": len(raw_predictions),
        "metrics": {name: {"mean": sum(values) / len(values) if values else None} for name, values in metrics.items()},
        "row_scores": row_scores,
        "predictions_sha256": hashlib.sha256("\n".join(raw_predictions).encode("utf-8")).hexdigest(),
    }
    save_json(summary_path, summary)
    print(f"Saved {display_name}: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional YAML run config; command-line options override config values.")
    parser.add_argument("--model", choices=MODELS, action="append", help="Repeat to run multiple baseline models; defaults to all released models.")
    parser.add_argument("--limit", type=int, help="Run only the first N test examples (for a smoke run).")
    parser.add_argument("--output-dir", type=Path, help="Output root; default: results.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--evaluate-only", action="store_true", help="Score complete saved predictions without calling model APIs.")
    mode.add_argument("--restart", action="store_true", help="Discard saved predictions and regenerate this run.")
    args = parser.parse_args()

    config = {}
    if args.config:
        import yaml

        config_path = args.config if args.config.is_absolute() else ROOT / args.config
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            parser.error("The YAML config must contain a mapping.")
    if args.model is None:
        args.model = config.get("models")
    if args.limit is None:
        args.limit = config.get("limit")
    if args.output_dir is None:
        configured_output = config.get("output_dir", "results")
        args.output_dir = Path(configured_output)
        if not args.output_dir.is_absolute():
            args.output_dir = ROOT / args.output_dir

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    dataset, revision = load_data(args.limit)
    output_root = args.output_dir.resolve()
    selected_models = args.model or MODELS
    for model_name in selected_models:
        run_model(model_name, dataset, revision, output_root, args)


if __name__ == "__main__":
    main()
