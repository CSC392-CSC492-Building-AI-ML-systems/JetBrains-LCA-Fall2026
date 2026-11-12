"""Synthetic short-ID edit smoke test: the same prompt with and without a token mask."""

import argparse
import ast
import importlib.metadata
import json
import sys
from pathlib import Path

import run_edit_format_pilot as pilot


SEEDS = (0, 1, 2)
ARMS = ("unconstrained", "constrained")
MAX_TOKENS = 192
TEMPERATURE = 0.2
MODEL = pilot.BENCHMARK / ".venv/mistral-model"
DEPS = pilot.BENCHMARK / ".venv/constrained-deps"

# Each tuple is (name, failure report, failing files, allowed blocks, reference files).
# Blocks are (relative path, first line, number of lines). References are for scoring only.
FIXTURES = (
    ("arithmetic", "test_total: total(2, 3) returned -1; expected 5. Add both inputs.",
     {"calc.py": "def total(a, b):\n    return a - b\n"}, [("calc.py", 2, 1)],
     {"calc.py": "def total(a, b):\n    return a + b\n"}),
    ("two_blocks", "test_limits: minimum must be 1 and maximum must be 9; both assertions failed.",
     {"bounds.py": "MINIMUM = 0\n\nMAXIMUM = 10\n"}, [("bounds.py", 1, 1), ("bounds.py", 3, 1)],
     {"bounds.py": "MINIMUM = 1\n\nMAXIMUM = 9\n"}),
    ("omit_unchanged", "test_count: COUNT should be 3, not 2. LABEL is correct and should stay unchanged.",
     {"values.py": "COUNT = 2\nLABEL = 'ready'\n"}, [("values.py", 1, 1), ("values.py", 2, 1)],
     {"values.py": "COUNT = 3\nLABEL = 'ready'\n"}),
    ("delete_block", "Lint: remove the unused TEMP flag entirely; keep COUNT.",
     {"flags.py": "TEMP = False\nCOUNT = 4\n"}, [("flags.py", 1, 1)],
     {"flags.py": "COUNT = 4\n"}),
    ("quoted_text", 'The banner must return the exact string She said "hi". Keep the function.',
     {"banner.py": "def banner():\n    return 'She said hi'\n"}, [("banner.py", 2, 1)],
     {"banner.py": "def banner():\n    return 'She said \"hi\"'\n"}),
    ("noop", "All checks passed; the configured status is correct. No repair is needed.",
     {"status.py": "STATUS = 'ready'\n"}, [("status.py", 1, 1)],
     {"status.py": "STATUS = 'ready'\n"}),
)


def blocks_from(files, locations):
    return [
        {"path": path, "start": start, "count": count,
         "old": "".join(files[path].splitlines(keepends=True)[start - 1:start - 1 + count])}
        for path, start, count in locations
    ]


def schema_for(block_count):
    if block_count < 1:
        raise ValueError("An edit request needs at least one block")
    return {"type": "object",
            "properties": {f"B{i}": {"type": "string"} for i in range(block_count)},
            "additionalProperties": False}


def grammar_for(block_count):
    from llguidance import LLMatcher

    # SentencePiece has tokens starting with a space; accept one before the object.
    return LLMatcher.grammar_from_lark(
        'start: " "? answer\nanswer: %json ' + json.dumps(schema_for(block_count))
    )


def prompt_for(failure, blocks):
    sections = "\n\n".join(
        f"Block B{i} in {block['path']} at line {block['start']}:\n{block['old']}"
        for i, block in enumerate(blocks)
    )
    return (
        f"Repair the code. Check output: {failure}\n\n"
        "Return only one JSON object mapping changed block IDs to their full replacement text. "
        "Keep keys in block order and omit unchanged blocks. Return {} for no changes. "
        'Use "" to delete an entire block. Escape newlines and quotes as JSON. '
        "Keep indentation. No Markdown or explanations.\n\n"
        + sections
    )


def parse_answer(response, blocks):
    """Validate again after sampling; unlike the old parser, no Markdown salvage."""
    try:
        edits = json.loads(response, object_pairs_hook=pilot.unique_keys)
    except json.JSONDecodeError as problem:
        raise ValueError("invalid_json") from problem
    allowed = schema_for(len(blocks))["properties"]
    if not isinstance(edits, dict) or not edits.keys() <= allowed.keys():
        raise ValueError("invalid_schema")
    if list(edits) != [key for key in allowed if key in edits]:
        raise ValueError("out_of_order_blocks")
    # Reuse the existing validated replacements and final-newline normalization.
    return pilot.parse_sparse(json.dumps({"edits": edits}), blocks)


class TokenConstraint:
    """Constrain the completion only, leaving MLX's prompt prefill untouched."""

    def __init__(self, matcher, prompt_length, eos):
        self.matcher = matcher
        self.seen = prompt_length
        self.eos = eos

    def __call__(self, tokens, logits):
        import mlx.core as mx
        import numpy as np

        if len(tokens) < self.seen or len(tokens) > self.seen + 1:
            raise RuntimeError("Unexpected MLX decoding step")
        if len(tokens) == self.seen + 1:
            token = tokens[-1].item()
            self.seen += 1
            if token == self.eos:
                if not self.matcher.is_accepting():
                    raise RuntimeError("Premature EOS")
            elif not self.matcher.consume_token(token):
                raise RuntimeError(f"Grammar rejected a sampled token: {self.matcher.get_error()}")
        if self.matcher.is_accepting():
            # Stop at the first complete object. MLX may calculate one lookahead
            # after sampling EOS; this keeps that extra step well-defined.
            allowed = mx.arange(logits.shape[-1]) == self.eos
        else:
            bias = np.frombuffer(self.matcher.compute_logit_bias(), dtype=np.uint8)
            if self.matcher.is_error() or bias.size != logits.shape[-1] or not bias.any():
                raise RuntimeError(f"Empty or incompatible grammar mask: {self.matcher.get_error()}")
            if bias[self.eos]:
                raise RuntimeError("Grammar permits EOS before the object is complete")
            allowed = mx.array(bias) != 0
        return mx.where(allowed, logits, -float("inf"))


def score_answer(response, finish, files, blocks, expected):
    try:
        if finish != "stop":
            raise ValueError("output_token_limit")
        replacements = parse_answer(response, blocks)
        valid = True
        error = None
    except ValueError as problem:
        replacements = [b["old"] for b in blocks]
        valid = False
        error = str(problem)
    patched = {**files, **pilot.apply_edits(files, blocks, replacements)}
    try:
        expected_ast = {p: ast.dump(ast.parse(text)) for p, text in expected.items()}
        patched_ast = {p: ast.dump(ast.parse(text)) for p, text in patched.items()}
        syntax = valid
        exact = valid and patched_ast == expected_ast
    except SyntaxError:
        syntax = exact = False
    return {"format_valid": valid, "error": error, "nonempty": patched != files,
            "syntax_valid": syntax, "reference_ast_match": exact,
            "changed_blocks": sum(b["old"] != r for b, r in zip(blocks, replacements))}


def summarize(rows):
    groups = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    totals = {}
    expected = len(FIXTURES) * len(SEEDS)
    for arm, group in groups.items():
        totals[arm] = {"scored": len(group), "expected": expected, **{
            key: sum(row[key] for row in group)
            for key in ("format_valid", "nonempty", "syntax_valid", "reference_ast_match")
        }}
    complete = all(len(group) == expected for group in groups.values())
    c = groups["constrained"]
    gate = None if not complete else (
        all(row["format_valid"] and row["syntax_valid"] for row in c)
        and sum(row["reference_ast_match"] for row in c) >= 12
        and all(any(row["fixture"] == f and row["reference_ast_match"] for row in c)
                for f in ("two_blocks", "omit_unchanged"))
    )
    return {"complete": complete, "synthetic_gate_passed": gate, "arms": totals, "cases": rows,
            "scope": "Handwritten synthetic fixtures only; no project or generated code executed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=pilot.BENCHMARK / "out/constrained-edit-smoke")
    args = parser.parse_args()
    if DEPS.is_dir():
        sys.path.insert(0, str(DEPS))
    from llguidance import LLMatcher, LLTokenizer
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    previous = pilot.read_json(
        pilot.BENCHMARK / "out/edit-format-pilot/0c6e365def079719ceb5/run_metadata.json"
    )
    for name, sha in previous["model_hashes"].items():
        if pilot.file_sha(MODEL / name) != sha:
            raise ValueError(f"Model identity mismatch: {name}")
    versions = {name: importlib.metadata.version(name) for name in [*previous["packages"], "llguidance"]}
    if any(versions[name] != version for name, version in previous["packages"].items()):
        raise ValueError("Model runtime differs from prior pilot")
    manifest = {"protocol": 1, "fixtures_sha256": pilot.digest(FIXTURES), "seeds": SEEDS, "arms": ARMS,
                "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "packages": versions,
                "model_hashes": previous["model_hashes"], "source_sha256": pilot.file_sha(Path(__file__)),
                "helper_sha256": pilot.file_sha(Path(pilot.__file__))}
    folder = args.out / pilot.digest(manifest)[:20]
    pilot.save_json(folder / "run_metadata.json", manifest)
    print(f"Synthetic smoke: {folder}; {len(FIXTURES) * len(SEEDS) * len(ARMS)} answers", flush=True)
    model, tokenizer = load(str(MODEL))
    eos, = tokenizer.eos_token_ids
    ll_tokenizer = LLTokenizer(tokenizer.backend_tokenizer.to_str(), eos_token=eos)
    rows = []
    for i, (name, failure, files, locations, expected) in enumerate(FIXTURES):
        blocks = blocks_from(files, locations)
        prompt = prompt_for(failure, blocks)
        tokens = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                add_generation_prompt=True, tokenize=True)
        grammar = grammar_for(len(blocks))
        for seed in SEEDS:
            for arm in ARMS:
                path = folder / "examples" / name / str(seed) / f"{arm}.json"
                identity = {"run": folder.name, "fixture": name, "seed": seed, "arm": arm}
                if path.exists():
                    answer, _ = pilot.read_checked(path)
                    if answer["identity"] != identity or answer["prompt"] != prompt:
                        raise ValueError("Synthetic prediction identity mismatch")
                else:
                    processors = []
                    if arm == "constrained":
                        matcher = LLMatcher(ll_tokenizer, grammar)
                        processors = [TokenConstraint(matcher, len(tokens), eos)]
                    response, finish, count = "", None, 0
                    mx.random.seed(seed + i)
                    for item in stream_generate(model, tokenizer, tokens, max_tokens=MAX_TOKENS,
                                                 sampler=make_sampler(temp=TEMPERATURE),
                                                 logits_processors=processors):
                        response += item.text
                        finish, count = item.finish_reason, item.generation_tokens
                    answer = {"identity": identity, "prompt": prompt, "response": response,
                              "finish_reason": finish, "generation_tokens": count}
                    pilot.save_json(path, {**answer, "payload_sha256": pilot.digest(answer)})
                score = score_answer(answer["response"], answer["finish_reason"], files, blocks, expected)
                rows.append({**identity, "generation_tokens": answer["generation_tokens"], **score})
                pilot.save_json(folder / "metrics.json", summarize(rows))
                print(f"{name}/{seed}/{arm}: {score['error'] or 'valid'} "
                      f"reference_ast={score['reference_ast_match']}", flush=True)
    print(json.dumps({k: v for k, v in summarize(rows).items() if k != "cases"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
