import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from metrics import score_one_model
from utils.files_utils import load_config
from utils.scorer import OptionsScoringModel


DATASET_NAME = "JetBrains-Research/lca-module-summarization"
SPLIT = "test"


def main():
    parser = argparse.ArgumentParser(description="Evaluate one Module Summarization baseline directory with CompScore.")
    parser.add_argument("--config", default="replication_configs/config_compscore_full.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    hf_api_key = config.get("hf_api_key")
    tokenizer = AutoTokenizer.from_pretrained(config["hf_tokenizer_checkpoint"], token=hf_api_key)
    dataset = load_dataset(DATASET_NAME, token=hf_api_key)[SPLIT]

    limit = config.get("limit")
    if limit is not None:
        dataset = dataset.select(range(min(int(limit), len(dataset))))

    scorer = OptionsScoringModel(config["scorer_model_name"], config["device"])
    scores = score_one_model(
        scorer,
        dataset,
        config["prediction_dir"],
        config["max_context_toks"],
        tokenizer,
        use_pbar=True,
    )

    output = {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "prediction_dir": config["prediction_dir"],
        "scorer_model_name": config["scorer_model_name"],
        "max_context_toks": config["max_context_toks"],
        "num_examples": len(scores),
        "compscore_mean": float(np.mean(scores)),
        "compscore_scores": scores,
    }

    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "compscore_scores"}, indent=2))


if __name__ == "__main__":
    main()
