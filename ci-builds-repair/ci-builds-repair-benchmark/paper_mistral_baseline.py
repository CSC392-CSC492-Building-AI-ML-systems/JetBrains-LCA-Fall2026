"""Paper-style oracle-block repair with Mistral-7B-Instruct-v0.2.

The gold diff supplies *locations and old text only*. Gold replacement lines
are never included in the prompt. This is an approximate reproduction because
the authors' prompt assembly implementation and decoding settings are absent.
"""

import json
import re
from pathlib import Path


MODEL_ID = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"


def relevant_logs(logs):
    """Extract the paper's seven-line window around the first 'error'."""
    parts = []
    for item in logs if isinstance(logs, list) else [logs]:
        if isinstance(item, dict):
            parts.extend(str(value) for value in item.values() if isinstance(value, str))
        else:
            parts.append(str(item))
    lines = "\n".join(parts).splitlines()
    position = next((i for i, line in enumerate(lines) if "error" in line.lower()), None)
    if position is None:
        return "\n".join(lines[-7:])
    return "\n".join(lines[max(0, position - 3):position + 4])


def oracle_blocks(diff, repository):
    """Read only old-side hunk spans from the failing checkout."""
    blocks, path = [], None
    lines = diff.splitlines()
    for line in lines:
        if line.startswith("+++ "):
            name = line[4:].split("\t", 1)[0]
            if name == "/dev/null":
                path = None
                continue
            name = name.removeprefix("b/")
            candidate = (Path(repository) / name).resolve()
            if not candidate.is_relative_to(Path(repository).resolve()) or not candidate.is_file():
                path = None
                continue
            path = name
        elif line.startswith("@@ ") and path:
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", line)
            if not match:
                raise ValueError(f"Invalid diff hunk header: {line}")
            start, count = int(match[1]), int(match[2] or 1)
            if count == 0:
                continue
            content = (Path(repository) / path).read_text(encoding="utf-8").splitlines(keepends=True)
            old = "".join(content[start - 1:start - 1 + count])
            if len(content[start - 1:start - 1 + count]) != count:
                raise ValueError(f"Hunk exceeds source file: {path}:{start}")
            blocks.append({"path": path, "start": start, "count": count, "old": old})
    if not blocks:
        raise ValueError("No editable source blocks in the gold diff")
    return blocks


def make_prompt(datapoint, blocks):
    sections = []
    for index, block in enumerate(blocks):
        tag = f"{block['path']}#L{block['start']}#B{index}"
        sections.append(f"[start of {tag}]\n{block['old'].rstrip()}\n[end of {tag}]")
    instruction = f"Fix CI in order for tests to pass. Relevant logs: {relevant_logs(datapoint['logs'])}"
    return (instruction + "\n\nEdit the following code blocks. Return every block exactly once with its "
            "start and end markers; put only replacement code between them. Do not add commentary.\n\n"
            + "\n\n".join(sections))


def parse_response(response, blocks):
    replacements = []
    for index, block in enumerate(blocks):
        tag = f"{block['path']}#L{block['start']}#B{index}"
        pattern = re.escape(f"[start of {tag}]") + r"\s*\n(.*?)\n" + re.escape(f"[end of {tag}]")
        matches = re.findall(pattern, response, flags=re.DOTALL)
        if len(matches) != 1:
            raise ValueError(f"Expected one replacement for {tag}; got {len(matches)}")
        replacements.append(matches[0] + "\n")
    return replacements


def apply_blocks(repository, blocks, replacements):
    grouped = {}
    for block, replacement in zip(blocks, replacements):
        grouped.setdefault(block["path"], []).append((block, replacement))
    updated = {}
    for name, edits in grouped.items():
        target = Path(repository) / name
        original = target.read_text(encoding="utf-8")
        lines = original.splitlines(keepends=True)
        last_start = len(lines) + 1
        for block, replacement in sorted(edits, key=lambda pair: pair[0]["start"], reverse=True):
            start, count = block["start"], block["count"]
            if start + count > last_start or "".join(lines[start - 1:start - 1 + count]) != block["old"]:
                raise ValueError(f"Overlapping or changed block: {name}:{start}")
            lines[start - 1:start - 1 + count] = replacement.splitlines(keepends=True)
            last_start = start
        updated[target] = "".join(lines)
    for target, content in updated.items():
        target.write_text(content, encoding="utf-8")


class MistralRepair:
    def __init__(self, model_path, seed=0, temperature=0.2):
        from mlx_lm import load
        self.model, self.tokenizer = load(str(model_path))
        self.seed = seed
        self.temperature = temperature

    def __call__(self, datapoint, repo_path, repo=None, out_folder=None):
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        mx.random.seed(self.seed + int(datapoint["id"]))
        blocks = oracle_blocks(datapoint["diff"], repo_path)
        prompt = make_prompt(datapoint, blocks)
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        response = generate(self.model, self.tokenizer, prompt=formatted, max_tokens=2048,
                            sampler=make_sampler(temp=self.temperature), verbose=False)
        record = {"id": datapoint["id"], "model": MODEL_ID, "seed": self.seed,
                  "effective_seed": self.seed + int(datapoint["id"]),
                  "temperature": self.temperature, "prompt": prompt, "response": response}
        output = Path(out_folder) / f"model_generation_{datapoint['id']}.json"
        try:
            replacements = parse_response(response, blocks)
            apply_blocks(repo_path, blocks, replacements)
            record["status"] = "applied"
        except ValueError as error:
            # A malformed answer is a failed repair, not a reason to stop the run.
            record["status"] = "unapplied"
            record["error"] = str(error)
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
