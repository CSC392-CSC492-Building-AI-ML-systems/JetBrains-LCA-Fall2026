import argparse
import json
import logging
import os
from pathlib import Path
from statistics import mean

from datasets import load_dataset
from openai import OpenAI
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from utils.api_generation import gpt_generation
from utils.context_utils import collect_good_context, trim_context
from utils.files_utils import load_config


DATASET_NAME = "JetBrains-Research/lca-module-summarization"
SPLIT = "test"
STRATEGY_NAME = "baseline_fixed_relevant_context"


def load_local_env(path=".env.local"):
    if not os.path.exists(path):
        return
    with open(path, "r") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                os.environ.setdefault(key, value)


def get_optional_secret(config, config_key, env_keys):
    value = config.get(config_key)
    if value and value != "YOUR_KEY_HERE":
        return value
    for env_key in env_keys:
        value = os.environ.get(env_key)
        if value:
            return value
    return None


def prepare_code_context(row, max_context_toks, tokenizer):
    context = collect_good_context(row)
    if max_context_toks is None:
        return context
    return trim_context(context, tokenizer, max_context_toks)


def generate_one(row, code_context, client, model_name):
    intent = row["intent"]
    filename = row["docfile_name"]

    prompt = "I have code collected from one or more files joined into one string. "
    prompt += f"Using the code generate text for {filename} file with documentation about {intent}.\n\n"
    prompt += f"My code:\n\n{code_context}"
    prompt += (
        f"\n\n\n\nAs answer return text for {filename} file about {intent}. "
        "Do not return the instruction how to make documentation, return only documentation itself."
    )

    return gpt_generation(client, prompt, model_name)


def rouge_l_f1(prediction, reference):
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    previous = [0] * (len(ref_tokens) + 1)
    for pred_token in pred_tokens:
        current = [0]
        for idx, ref_token in enumerate(ref_tokens, start=1):
            if pred_token == ref_token:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current

    lcs = previous[-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def describe(values):
    if not values:
        return {"min": None, "mean": None, "max": None}
    sorted_values = sorted(values)
    return {
        "min": sorted_values[0],
        "mean": mean(sorted_values),
        "max": sorted_values[-1],
    }


def load_generation_config(config_path):
    config = load_config(config_path)
    config.setdefault("dataset_name", DATASET_NAME)
    config.setdefault("split", SPLIT)
    config.setdefault("strategy_name", STRATEGY_NAME)
    config.setdefault("max_context_toks", 2000)
    config.setdefault("summary_path", str(Path(config["save_dir"]) / "summary.json"))
    config.setdefault("skip_existing", True)
    return config


def select_dataset(config):
    hf_api_key = get_optional_secret(config, "hf_api_key", ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"])
    dataset = load_dataset(config["dataset_name"], token=hf_api_key)[config["split"]]
    limit = config.get("limit")
    if limit is not None:
        dataset = dataset.select(range(min(int(limit), len(dataset))))
    return dataset


def build_summary(config, dataset, save_dir):
    rows = []
    missing_prediction_indices = []

    for idx, row in enumerate(dataset):
        prediction_path = save_dir / f"{idx}.txt"
        if not prediction_path.exists():
            missing_prediction_indices.append(idx)
            continue

        prediction = prediction_path.read_text()
        target = row["target_text"] or ""
        rows.append(
            {
                "index": idx,
                "repo": row["repo"],
                "docfile_name": row["docfile_name"],
                "intent": row["intent"],
                "prediction_chars": len(prediction),
                "prediction_words": len(prediction.split()),
                "target_chars": len(target),
                "target_words": len(target.split()),
                "rouge_l_f1": rouge_l_f1(prediction, target),
            }
        )

    return {
        "dataset": config["dataset_name"],
        "split": config["split"],
        "model": config["model_name"],
        "strategy": config["strategy_name"],
        "max_context_toks": config.get("max_context_toks"),
        "metric_note": "ROUGE-L F1 is a local diagnostic only. The official LCA Module Summarization metric is CompScore.",
        "num_dataset_rows": len(dataset),
        "num_predictions": len(rows),
        "missing_prediction_indices": missing_prediction_indices,
        "rouge_l_f1": describe([row["rouge_l_f1"] for row in rows]),
        "prediction_chars": describe([row["prediction_chars"] for row in rows]),
        "prediction_words": describe([row["prediction_words"] for row in rows]),
        "target_chars": describe([row["target_chars"] for row in rows]),
        "target_words": describe([row["target_words"] for row in rows]),
        "rows": rows,
    }


def run(config):
    hf_api_key = get_optional_secret(config, "hf_api_key", ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"])
    api_key = get_optional_secret(config, "api_key", ["OPENAI_API_KEY"])
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set in .env.local or the environment.")

    save_dir = Path(config["save_dir"])
    logs_dir = Path(config.get("logs_dir", save_dir / "logs"))
    summary_path = Path(config["summary_path"])
    save_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = config["model_name"]
    logging.basicConfig(
        filename=logs_dir / f"openai_gen_{model_name}.log",
        encoding="utf-8",
        level=logging.INFO,
    )

    logging.info("Loading dataset")
    dataset = select_dataset(config)
    logging.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["hf_tokenizer_checkpoint"], token=hf_api_key)
    client = OpenAI(api_key=api_key)

    skip_existing = bool(config.get("skip_existing", True))
    for row_idx, row in tqdm(enumerate(dataset), total=len(dataset), desc="Generation"):
        prediction_path = save_dir / f"{row_idx}.txt"
        if skip_existing and prediction_path.exists():
            continue

        code_context = prepare_code_context(row, config.get("max_context_toks"), tokenizer)
        generated = generate_one(row, code_context, client, model_name)
        prediction_path.write_text(generated)

    summary = build_summary(config, dataset, save_dir)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in summary if key != "rows"}, indent=2))


def main():
    load_local_env()

    parser = argparse.ArgumentParser(description="Run the Module Summarization baseline replication.")
    parser.add_argument("--config", default="replication_configs/config_openai_smoke.yaml", help="Path to a YAML config file.")
    args = parser.parse_args()

    run(load_generation_config(args.config))


if __name__ == "__main__":
    main()
