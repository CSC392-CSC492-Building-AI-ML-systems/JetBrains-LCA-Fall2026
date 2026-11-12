"""Offline reference-patch metrics and syntax checks; never executes project code.

These are proxies for repair quality, not CI Pass@1 or proof of correctness.
"""

import ast
import difflib
import re
import shlex
from collections import Counter
from pathlib import PurePosixPath

from retrieval import source_lines


def safe_path(value):
    value = value.removeprefix("a/").removeprefix("b/")
    path = PurePosixPath(value)
    if value == "/dev/null":
        return None
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("Unsafe diff path")
    return value


def diff_blocks(diff):
    """Parse old and new hunk sides; prediction must receive only old-side fields."""
    blocks, old_path, new_path, current, last_kind = [], None, None, None, None
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git "):
            current, old_path, new_path = None, None, None
        elif line.startswith("--- "):
            old_path = safe_path(shlex.split(line[4:].strip())[0])
            current = None
        elif line.startswith("+++ "):
            new_path = safe_path(shlex.split(line[4:].strip())[0])
            if old_path and new_path and old_path != new_path:
                raise ValueError("Renamed files are not supported by the oracle block editor")
        elif line.startswith("@@ "):
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if match is None or not (old_path or new_path):
                raise ValueError("Malformed hunk")
            current = {"path": new_path or old_path, "start": int(match[1]),
                       "count": int(match[2] or 1), "new_count": int(match[4] or 1),
                       "old": "", "new": ""}
            blocks.append(current)
        elif current is not None:
            if line.startswith("\\ No newline"):
                for side in (("old", "new") if last_kind == " " else
                             ("old",) if last_kind == "-" else ("new",)):
                    current[side] = current[side].removesuffix("\n")
            elif line[:1] in (" ", "-", "+"):
                last_kind = line[0]
                if last_kind in (" ", "-"):
                    current["old"] += line[1:]
                if last_kind in (" ", "+"):
                    current["new"] += line[1:]
    if not blocks:
        raise ValueError("No text hunks")
    for block in blocks:
        if len(source_lines(block["old"])) != block["count"] or len(source_lines(block["new"])) != block["new_count"]:
            raise ValueError("Hunk length mismatch")
    return blocks


def prediction_blocks(blocks):
    return [{key: block[key] for key in ("path", "start", "count", "old")} for block in blocks]


def apply_edits(files, blocks, replacements):
    if len(blocks) != len(replacements):
        raise ValueError("Replacement count mismatch")
    edits = {}
    for block, replacement in zip(blocks, replacements):
        edits.setdefault(block["path"], []).append((block, replacement))
    output = {}
    for path, changes in edits.items():
        lines = source_lines(files.get(path, ""))
        boundary = len(lines) + 1
        for block, replacement in sorted(changes, key=lambda pair: pair[0]["start"], reverse=True):
            offset = block["start"] - 1 if block["count"] else block["start"]
            if offset < 0 or offset + block["count"] > boundary:
                raise ValueError("Overlapping/out-of-range hunk")
            if "".join(lines[offset:offset + block["count"]]) != block["old"]:
                raise ValueError("Gold old-side text differs from failing snapshot")
            lines[offset:offset + block["count"]] = source_lines(replacement)
            boundary = offset
        output[path] = "".join(lines)
    return output


def edit_tokens(before, after):
    result = Counter()
    for path, text in after.items():
        for line in difflib.unified_diff(source_lines(before.get(path, "")), source_lines(text), n=0):
            if line.startswith(("+++", "---", "@@")) or line[:1] not in ("+", "-"):
                continue
            result.update((path, line[0], token) for token in re.findall(r"\w+|[^\w\s]", line[1:]))
    return result


def parses(text):
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError):
        return False


def score(files, blocks, predicted, retrieval, packed):
    gold = apply_edits(files, blocks, [block["new"] for block in blocks])
    predicted = {path: predicted.get(path, files.get(path, "")) for path in gold}
    gold_tokens, predicted_tokens = edit_tokens(files, gold), edit_tokens(files, predicted)
    common = sum((gold_tokens & predicted_tokens).values())
    precision = common / sum(predicted_tokens.values()) if predicted_tokens else 0.0
    recall = common / sum(gold_tokens.values()) if gold_tokens else 0.0
    exact = all(predicted[path] == text for path, text in gold.items())
    python_paths = [path for path in gold if path.endswith(".py") and parses(files.get(path, ""))]
    regressions = [path for path in python_paths if not parses(predicted[path])]
    gold_paths = set(gold)
    def coverage(chunks):
        hits = len(gold_paths & {chunk["path"] for chunk in chunks})
        return {"file_recall": hits / len(gold_paths), "all_files": hits == len(gold_paths)}
    return {"patch_exact_match": exact, "edit_token_precision": precision, "edit_token_recall": recall,
            "edit_token_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "nonempty_patch": any(predicted[path] != files.get(path, "") for path in gold),
            "syntax_eligible_files": len(python_paths), "syntax_regression_files": regressions,
            "syntax_preserved": not regressions,
            "retrieval_before_packing": coverage(retrieval), "retrieval_after_packing": coverage(packed)}
