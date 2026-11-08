"""Tiny real Git repository and label-guarded rows; never uses network or API."""

from pathlib import Path

from git import Actor, Repo

from bug_localization.rag.model import ModelResponse
from bug_localization.rag.retrieval import PREDICTION_FIELDS


class GuardedRow(dict):
    def __getitem__(self, key):
        if key not in PREDICTION_FIELDS:
            raise AssertionError(f"Prediction touched forbidden field {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key not in PREDICTION_FIELDS:
            raise AssertionError(f"Prediction touched forbidden field {key}")
        return super().get(key, default)


def make_repository(folder):
    folder = Path(folder)
    folder.mkdir(parents=True)
    repo = Repo.init(folder)
    actor = Actor("Offline fixture", "fixture@example.invalid")
    content = {"z.py": "def beta():\n    return 'beta failure'\n", "a.py": "alpha = 'alpha failure'\n",
               "empty.py": "", "tests/excluded.py": "FORBIDDEN_TEST = True\n",
               "README.md": "FORBIDDEN_NON_PYTHON\n", "upper.PY": "FORBIDDEN_EXTENSION\n"}
    for path, text in content.items():
        dest = folder / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8", newline="")
    # Invalid UTF-8 is rejected by the shared reader if it is a candidate; an
    # excluded test file must never be decoded or change the candidate universe.
    (folder / "tests" / "binary.py").write_bytes(b"\xff")
    repo.index.add([*content, "tests/binary.py"])
    base = repo.index.commit("buggy base", author=actor, committer=actor,
                             author_date="2020-01-01T00:00:00+0000", commit_date="2020-01-01T00:00:00+0000")
    (folder / "a.py").write_text("FIXED_HEAD_MUST_NOT_APPEAR\n", encoding="utf-8", newline="")
    repo.index.add(["a.py"])
    head = repo.index.commit("later fix", author=actor, committer=actor,
                             author_date="2020-01-02T00:00:00+0000", commit_date="2020-01-02T00:00:00+0000")
    rows = [{"repo_owner": "fixture", "repo_name": "repo", "base_sha": base.hexsha,
             "issue_url": f"https://example.invalid/issues/{i}", "issue_title": title,
             "issue_body": "Failure in the alpha/beta source.", "changed_files": labels,
             "head_sha": head.hexsha, "diff": "FORBIDDEN_DIFF"}
            for i, (title, labels) in enumerate([
                ("alpha failure", ["a.py", "tests/excluded.py"]), ("beta failure", ["z.py"])])]
    repo.close()
    return folder, rows


class MockClient:
    """Respond solely from supplied input; this object cannot access labels."""
    def __init__(self):
        self.calls = []

    def generate(self, messages):
        import json
        self.calls.append(messages)
        title = json.loads(messages[1]["content"])["issue_title"]
        paths = ["z.py", "a.py", "empty.py"] if title.startswith("beta") else ["a.py", "z.py", "empty.py"]
        return ModelResponse(json.dumps({"ranked_files": paths}), latency_seconds=.001,
                             usage={"prompt_tokens": 123, "completion_tokens": 12}, finish_reason="stop")
