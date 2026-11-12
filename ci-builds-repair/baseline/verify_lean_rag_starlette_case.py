"""Check Starlette case 144's failing literal and reference AST without executing code."""

import ast
import json

import run_qwen_lean_rag as lean
from verify_lean_rag_black_case import RUN

pilot = lean.pilot
CASE = 144
FILE = "tests/test_responses.py"
TEST = "test_file_response_with_pathsend"


def inspect(text, reference_tree, expected_scope):
    result = {"scope_matches_reference": False, "literal_error": None,
              "file_ast_matches_reference": False, "send_call_count_access": False}
    try:
        tree = ast.parse(text)
        function, = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == TEST]
        scope, = [n.value.args[0] for n in ast.walk(function) if isinstance(n, ast.Await)
                  and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                  and n.value.func.id == "app" and n.value.args]
        result["file_ast_matches_reference"] = ast.dump(tree) == ast.dump(reference_tree)
        result["send_call_count_access"] = any(isinstance(n, ast.Attribute) and n.attr == "call_count"
                                               and isinstance(n.value, ast.Name) and n.value.id == "send"
                                               for n in ast.walk(function))
        # Evaluate just the literal, not the test, app, or generated Python.
        result["scope_matches_reference"] = ast.literal_eval(scope) == expected_scope
    except (SyntaxError, ValueError, TypeError) as exc:
        result["literal_error"] = f"{type(exc).__name__}: {exc}"
    return result


def main():
    from datasets import Dataset

    manifest = pilot.read_json(RUN / "run_metadata.json")
    if RUN.name != pilot.digest(manifest)[:20] or not pilot.read_json(RUN / "metrics.json")["complete"]:
        raise ValueError("Expected a complete lean-retrieval run")
    arrow, = [p for p in (pilot.BENCHMARK / ".venv/dataset-cache").rglob("*test.arrow")
              if pilot.file_sha(p) == manifest["dataset_sha256"]]
    row, = [r for r in Dataset.from_file(str(arrow)) if r["id"] == CASE]
    if row["sha_fail"] != manifest["failing_shas"][str(CASE)] or row["changed_files"] != [FILE]:
        raise ValueError("Unexpected Starlette example")
    gold = pilot.diff_blocks(row["diff"])
    blocks = pilot.prediction_blocks(gold)
    files = pilot.target_files(row, blocks, lean.CACHE)
    reference = pilot.apply_edits(files, blocks, [b["new"] for b in gold])[FILE]
    tree = ast.parse(reference)
    scope = {"type": "http", "method": "get", "extensions": {"http.response.pathsend": {}}}
    results = {"failed": inspect(files[FILE], tree, scope), "reference": inspect(reference, tree, scope)}
    error = results["failed"]["literal_error"]
    if not error or error not in row["logs"][0]["log"] or not results["reference"]["scope_matches_reference"]:
        raise ValueError("Failed/reference controls do not match the logged literal error")
    evidence, evidence_sha = pilot.read_checked(RUN / "evidence" / f"{CASE}.json")
    if evidence["sha_fail"] != row["sha_fail"]:
        raise ValueError("Evidence revision mismatch")
    for seed in manifest["seeds"]:
        for arm in lean.ARMS:
            answer, _ = pilot.read_checked(RUN / "examples" / str(CASE) / str(seed) / f"{arm}.json")
            if answer["identity"] != {"run": RUN.name, "id": CASE, "seed": seed, "arm": arm,
                                      "evidence_sha256": evidence_sha}:
                raise ValueError("Saved prediction identity mismatch")
            edits, _ = lean.dev.replacements_for(answer, blocks, "legacy_plain")
            text = pilot.apply_edits(files, blocks, edits)[FILE]
            results[f"{arm}/{seed}"] = inspect(text, tree, scope)
    report = {"source_run": RUN.name, "case": CASE, "file": FILE, "failing_literal_matches_log": True,
              "scope": "Literal and AST replay only; no tests, app, or generated code executed",
              "results": results}
    pilot.save_json(RUN / "case_144_literal.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
