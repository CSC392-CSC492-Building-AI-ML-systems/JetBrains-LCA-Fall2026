"""Reuse saved baseline answers when only the unused input budget changes.

For the baseline method, retrieval is empty and the prompt does not depend on
the context-window budget. A prediction is reusable only when its original
prompt fit that budget and every other run setting is identical.
"""

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rag"))
from run_benchmark import digest, save_json  # noqa: E402


def read_verified(path):
    value = json.loads(path.read_text())
    checksum = value.pop("payload_sha256", None)
    if checksum != digest(value):
        raise ValueError(f"Invalid prediction checksum: {path}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    old = json.loads((args.source / "run_metadata.json").read_text())
    new = json.loads((args.target / "run_metadata.json").read_text())
    allowed_changes = {"methods", "context_window"}
    for key in old.keys() | new.keys():
        if key not in allowed_changes and old.get(key) != new.get(key):
            raise ValueError(f"Incompatible run setting: {key}")
    if "baseline" not in old["methods"] or new["methods"] != ["baseline"]:
        raise ValueError("Expected baseline method in source and baseline-only target")
    if new["context_window"] < old["context_window"]:
        raise ValueError("Target input budget must not shrink")
    old_budget = old["context_window"] - old["max_tokens"] - 256
    copied = existing = skipped = 0
    for source in sorted((args.source / "examples").glob("*/seed_*/baseline/prediction.json")):
        relative = source.relative_to(args.source)
        target = args.target / relative
        value = read_verified(source)
        if value["identity"]["run"] != args.source.name:
            raise ValueError(f"Wrong source identity: {source}")
        if value["prompt_tokens"] > old_budget:
            if value["error"] != "oracle_prompt_exceeds_budget":
                raise ValueError(f"Old prompt exceeds budget but was generated: {source}")
            skipped += 1
            continue
        if value["error"] == "oracle_prompt_exceeds_budget":
            raise ValueError(f"Old prompt fit budget but was skipped: {source}")
        value["identity"] = {**value["identity"], "run": args.target.name}
        if target.exists():
            current = read_verified(target)
            compared = ("prompt", "prompt_tokens", "response", "finish_reason",
                        "generation_tokens", "packed_chunks", "replacements",
                        "status", "error")
            if any(current[k] != value[k] for k in compared):
                raise ValueError(f"Existing target prediction differs: {target}")
            existing += 1
        else:
            save_json(target, {**value, "payload_sha256": digest(value)})
            copied += 1
    record = {"source_run": args.source.name, "target_run": args.target.name,
              "reason": "baseline prompt and decoding unchanged; input budget increased",
              "copied": copied, "already_generated_and_matched": existing,
              "requires_new_inference": skipped}
    record_path = args.target / "prediction_reuse.json"
    if not record_path.exists():
        save_json(record_path, record)
    else:
        prior = json.loads(record_path.read_text())
        if any(prior[key] != record[key] for key in ("source_run", "target_run", "requires_new_inference")):
            raise ValueError("Existing reuse record describes another migration")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
