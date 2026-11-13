"""Paired synthetic smoke test for structured decisions and raw Python edits."""

import argparse
import ast
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path

import run_constrained_edit_smoke as previous


pilot = previous.pilot
SEEDS = previous.SEEDS
ARMS = previous.ARMS
MAX_TOKENS = previous.MAX_TOKENS
TEMPERATURE = previous.TEMPERATURE
FIXTURES = (*previous.FIXTURES, (
    "multiline", "test_summarize: summarize([2, 3]) returned 2; expected 5. Sum the values.",
    {"summary.py": "def summarize(numbers):\n    total = len(numbers)\n    return total\n"},
    [("summary.py", 2, 2)],
    {"summary.py": "def summarize(numbers):\n    total = sum(numbers)\n    return total\n"},
))
HEADER = re.compile(r"B(0|[1-9][0-9]*) (KEEP|DELETE|REPLACE)")


def grammar_for(count):
    from llguidance import LLMatcher

    if count < 1:
        raise ValueError("Expected at least one block")
    rules = ['start: " "? ' + " ".join(f"b{i}?" for i in range(count)) + ' "DONE"',
             r'op: " KEEP\n" | " DELETE\n" | " REPLACE\n" code_line+ "END\n"',
             # Match a complete raw code line, except the reserved END line.
             r'code_line: /(?:[^E\r\n][^\r\n]*|E(?:[^N\r\n][^\r\n]*)?|EN(?:[^D\r\n][^\r\n]*)?|END[^\r\n]+)?\n/']
    rules.extend(f'b{i}: "B{i}" op' for i in range(count))
    return LLMatcher.grammar_from_lark("\n".join(rules))


def base_indent(text):
    return os.path.commonprefix([re.match(r"[ \t]*", line)[0]
                                 for line in text.splitlines() if line.strip()])


def strip_indent(text, indent):
    return "".join(line.removeprefix(indent) if line.strip() else line
                   for line in text.splitlines(keepends=True))


def anchor_code(text, block):
    indent = base_indent(block["old"])
    anchored = "".join(indent + line if line.strip() else line
                       for line in text.splitlines(keepends=True))
    if block["old"] and not block["old"].endswith(("\r", "\n")):
        anchored = anchored.removesuffix("\n")
    return anchored


def prompt_for(failure, blocks):
    sections = []
    for i, b in enumerate(blocks):
        indent = base_indent(b["old"])
        sections.append(f"Block B{i} ({b['path']}, line {b['start']}):\n{strip_indent(b['old'], indent)}")
    return (
        f"Repair the code. Check output: {failure}\n\n"
        "For each changed block, write B<number> DELETE to remove it, or B<number> REPLACE "
        "followed by raw Python on subsequent lines, then END on its own line. "
        "Keep unchanged blocks omitted or use B<number> KEEP. Finish with DONE on its own line; "
        "if no changes are needed, return only DONE. Use ascending block numbers. "
        "The code shown is relative to each block's base indentation, which is restored "
        "automatically. Preserve relative indentation. Do not use JSON or Markdown.\n\n"
        + "\n\n".join(sections)
    )


def parse_answer(answer, blocks):
    lines = answer.removeprefix(" ").splitlines(keepends=True)
    if not lines or lines[-1] not in {"DONE", "DONE\n"} or "\0" in answer or "\r" in answer:
        raise ValueError("invalid_envelope")
    replacements = [b["old"] for b in blocks]
    last = -1
    i = 0
    while i < len(lines) - 1:
        match = HEADER.fullmatch(lines[i].removesuffix("\n"))
        if not match:
            raise ValueError("invalid_action")
        block_id, action = int(match[1]), match[2]
        if not last < block_id < len(blocks):
            raise ValueError("unknown_repeated_or_out_of_order_block")
        last = block_id
        i += 1
        if action == "DELETE":
            replacements[block_id] = ""
        elif action == "REPLACE":
            start = i
            while i < len(lines) - 1 and lines[i] != "END\n":
                i += 1
            if i == len(lines) - 1 or not "".join(lines[start:i]).strip():
                raise ValueError("empty_or_unterminated_replacement")
            replacements[block_id] = anchor_code("".join(lines[start:i]), blocks[block_id])
            i += 1
    return replacements


def evaluate(answer, files, blocks, expected):
    try:
        if answer["finish_reason"] != "stop":
            raise ValueError("output_token_limit")
        replacements = parse_answer(answer["response"], blocks)
    except ValueError as exc:
        return {"format_valid": False, "nonempty": False, "syntax_valid": False,
                "reference_ast_match": False, "error": str(exc)}
    patched = {**files, **pilot.apply_edits(files, blocks, replacements)}
    try:
        patched_ast = {path: ast.dump(ast.parse(text)) for path, text in patched.items()}
        expected_ast = {path: ast.dump(ast.parse(text)) for path, text in expected.items()}
        syntax = True
        exact = patched_ast == expected_ast
    except SyntaxError:
        syntax = exact = False
    return {"format_valid": True, "nonempty": patched != files, "syntax_valid": bool(syntax),
            "reference_ast_match": exact, "error": None}


def summarize(rows):
    n = len(FIXTURES) * len(SEEDS)
    groups = {arm: [x for x in rows if x["arm"] == arm] for arm in ARMS}
    totals = {arm: {"scored": len(g), **{key: sum(x[key] for x in g) for key in
              ("format_valid", "nonempty", "syntax_valid", "reference_ast_match")}} for arm, g in groups.items()}
    complete = all(len(g) == n for g in groups.values())
    constrained = groups["constrained"]
    gate = None if not complete else (
        all(x["format_valid"] and x["syntax_valid"] for x in constrained)
        and totals["constrained"]["reference_ast_match"] >= 14
        and all(any(x["fixture"] == name and x["reference_ast_match"] for x in constrained)
                for name in ("two_blocks", "omit_unchanged", "multiline", "noop"))
    )
    return {"complete": complete, "synthetic_gate_passed": gate, "arms": totals, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=pilot.BENCHMARK / "out/raw-edit-smoke")
    args = parser.parse_args()
    if previous.DEPS.is_dir():
        sys.path.insert(0, str(previous.DEPS))
    from llguidance import LLMatcher, LLTokenizer
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    baseline = pilot.read_json(pilot.BENCHMARK / "out/constrained-edit-smoke/1c5e1e46c44b3f21eac9/run_metadata.json")
    for name, sha in baseline["model_hashes"].items():
        if pilot.file_sha(previous.MODEL / name) != sha:
            raise ValueError(f"Changed model file: {name}")
    if {k: importlib.metadata.version(k) for k in baseline["packages"]} != baseline["packages"]:
        raise ValueError("Changed runtime")
    manifest = {"fixture_hash": pilot.digest(FIXTURES), "seeds": SEEDS, "arms": ARMS,
                "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "model_hashes": baseline["model_hashes"],
                "packages": baseline["packages"], "source_sha256": pilot.file_sha(Path(__file__)),
                "mask_source_sha256": pilot.file_sha(Path(previous.__file__))}
    folder = args.out / pilot.digest(manifest)[:20]
    pilot.save_json(folder / "run_metadata.json", manifest)
    print(f"Raw-code smoke: {folder}; {len(FIXTURES) * len(SEEDS) * len(ARMS)} answers", flush=True)
    model, tokenizer = load(str(previous.MODEL))
    eos, = tokenizer.eos_token_ids
    grammar_tokenizer = LLTokenizer(tokenizer.backend_tokenizer.to_str(), eos_token=eos)
    metrics = []
    for index, (name, failure, files, locations, expected) in enumerate(FIXTURES):
        blocks = previous.blocks_from(files, locations)
        prompt = prompt_for(failure, blocks)
        tokens = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                add_generation_prompt=True, tokenize=True)
        grammar = grammar_for(len(blocks))
        for seed in SEEDS:
            for arm in ARMS:
                identity = {"run": folder.name, "fixture": name, "seed": seed, "arm": arm}
                path = folder / "examples" / name / str(seed) / f"{arm}.json"
                if path.exists():
                    answer, _ = pilot.read_checked(path)
                    if answer["identity"] != identity or answer["prompt"] != prompt:
                        raise ValueError("Saved answer mismatch")
                else:
                    processors = ([previous.TokenConstraint(LLMatcher(grammar_tokenizer, grammar), len(tokens), eos)]
                                  if arm == "constrained" else [])
                    response, finish, count = "", None, 0
                    mx.random.seed(seed + index)
                    for piece in stream_generate(model, tokenizer, tokens, max_tokens=MAX_TOKENS,
                                                  sampler=make_sampler(temp=TEMPERATURE),
                                                  logits_processors=processors):
                        response += piece.text
                        finish, count = piece.finish_reason, piece.generation_tokens
                    answer = {"identity": identity, "prompt": prompt, "response": response,
                              "finish_reason": finish, "generation_tokens": count}
                    pilot.save_json(path, {**answer, "payload_sha256": pilot.digest(answer)})
                result = {**identity, "generation_tokens": answer["generation_tokens"],
                          **evaluate(answer, files, blocks, expected)}
                metrics.append(result)
                pilot.save_json(folder / "metrics.json", summarize(metrics))
                print(f"{name}/{seed}/{arm}: {result['error'] or 'valid'} "
                      f"syntax={result['syntax_valid']} reference_ast={result['reference_ast_match']}", flush=True)
    print(json.dumps({k: v for k, v in summarize(metrics).items() if k != "cases"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
