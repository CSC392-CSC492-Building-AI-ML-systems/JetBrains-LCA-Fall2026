"""Compare constrained raw edits on the same synthetic fixtures with Mistral and Qwen Coder."""

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import run_raw_edit_smoke as raw


pilot = raw.pilot
QWEN_ID = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
QWEN_REVISION = "019cc73c45c770444708a6dd8690c66243cc5c80"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mistral", type=Path, default=pilot.BENCHMARK / "out/raw-edit-smoke/2d9f6819a99991705adf")
    parser.add_argument("--qwen", type=Path, default=pilot.BENCHMARK / ".venv/qwen-coder-model")
    parser.add_argument("--out", type=Path, default=pilot.BENCHMARK / "out/raw-edit-model-comparison")
    args = parser.parse_args()
    if raw.previous.DEPS.is_dir():
        sys.path.insert(0, str(raw.previous.DEPS))
    from llguidance import LLMatcher, LLTokenizer
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    original = pilot.read_json(args.mistral / "run_metadata.json")
    if (args.mistral.name != pilot.digest(original)[:20]
        or original["fixture_hash"] != pilot.digest(raw.FIXTURES)
        or original["source_sha256"] != pilot.file_sha(Path(raw.__file__))
        or original["mask_source_sha256"] != pilot.file_sha(Path(raw.previous.__file__))):
        raise ValueError("Mistral protocol identity mismatch")
    if not pilot.read_json(args.mistral / "metrics.json")["complete"]:
        raise ValueError("Mistral run is incomplete")
    if {k: importlib.metadata.version(k) for k in original["packages"]} != original["packages"]:
        raise ValueError("Runtime differs from Mistral")
    expected_names = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
    qwen_hashes = {f.name: pilot.file_sha(f) for f in sorted(args.qwen.iterdir()) if f.is_file()}
    if not all(name in qwen_hashes for name in expected_names):
        raise ValueError("Qwen snapshot is incomplete")
    manifest = {"control_run": args.mistral.name, "model": QWEN_ID, "revision": QWEN_REVISION,
                "qwen_hashes": qwen_hashes, "fixture_hash": original["fixture_hash"],
                "seeds": raw.SEEDS, "max_tokens": raw.MAX_TOKENS, "temperature": raw.TEMPERATURE,
                "packages": original["packages"], "protocol_source": original["source_sha256"],
                "comparison_source": pilot.file_sha(Path(__file__))}
    folder = args.out / pilot.digest(manifest)[:20]
    pilot.save_json(folder / "run_metadata.json", manifest)
    print(f"Model comparison: {folder}", flush=True)
    model, tokenizer = load(str(args.qwen))
    eos = tokenizer.eos_token_id
    if eos not in tokenizer.eos_token_ids:
        raise ValueError("Qwen EOS mismatch")
    vocab_size = pilot.read_json(args.qwen / "config.json")["vocab_size"]
    backend = LLTokenizer(tokenizer.backend_tokenizer.to_str(), n_vocab=vocab_size, eos_token=eos)
    scores = []
    for index, (name, failure, files, locations, gold) in enumerate(raw.FIXTURES):
        blocks = raw.previous.blocks_from(files, locations)
        prompt = raw.prompt_for(failure, blocks)
        tokens = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                add_generation_prompt=True, tokenize=True)
        grammar = raw.grammar_for(len(blocks))
        for seed in raw.SEEDS:
            old, old_sha = pilot.read_checked(args.mistral / "examples" / name / str(seed) / "constrained.json")
            if old["identity"] != {"run": args.mistral.name, "fixture": name, "seed": seed, "arm": "constrained"} or old["prompt"] != prompt:
                raise ValueError("Saved Mistral answer mismatch")
            path = folder / "examples" / name / str(seed) / "qwen.json"
            identity = {"run": folder.name, "fixture": name, "seed": seed, "control_sha256": old_sha}
            if path.exists():
                answer, _ = pilot.read_checked(path)
                if answer["identity"] != identity or answer["prompt"] != prompt:
                    raise ValueError("Saved Qwen answer mismatch")
            else:
                processor = raw.previous.TokenConstraint(LLMatcher(backend, grammar), len(tokens), eos)
                response, finish, count = "", None, 0
                mx.random.seed(seed + index)
                for piece in stream_generate(model, tokenizer, tokens, max_tokens=raw.MAX_TOKENS,
                                              sampler=make_sampler(temp=raw.TEMPERATURE),
                                              logits_processors=[processor]):
                    response += piece.text
                    finish, count = piece.finish_reason, piece.generation_tokens
                answer = {"identity": identity, "prompt": prompt, "response": response,
                          "finish_reason": finish, "generation_tokens": count}
                pilot.save_json(path, {**answer, "payload_sha256": pilot.digest(answer)})
            scores.append({"fixture": name, "seed": seed,
                           "mistral": raw.evaluate(old, files, blocks, gold),
                           "qwen": raw.evaluate(answer, files, blocks, gold)})
            print(f"{name}/{seed}: Qwen {scores[-1]['qwen']}", flush=True)
            summary = {"complete": len(scores) == len(raw.FIXTURES) * len(raw.SEEDS),
                       "matched_cases": len(scores), "models": {
                model: {field: sum(row[model][field] for row in scores) for field in
                        ("format_valid", "nonempty", "syntax_valid", "reference_ast_match")}
                for model in ("mistral", "qwen")
            }, "cases": scores}
            pilot.save_json(folder / "metrics.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
