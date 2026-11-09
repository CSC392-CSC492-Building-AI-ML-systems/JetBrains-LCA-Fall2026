"""A distinct Git base/head plus an evidence-driven mocked model."""

import json
import re
from pathlib import Path

from git import Actor, Repo

from bug_localization.long_context.tests.fixtures import GuardedRow
from bug_localization.rag.model import ModelResponse


def make_repository(folder):
    folder = Path(folder)
    folder.mkdir(parents=True)
    repo = Repo.init(folder)
    files = {"router.py": "# crash entry point\nfrom symbols import resolve_alias\ndef route(value):\n    return resolve_alias(value)\n",
             "symbols.py": "def resolve_alias(value):\n    return value.target\n",
             "empty.py": "", "tests/excluded.py": "FORBIDDEN_TEST\n", "README.md": "FORBIDDEN_DOC\n"}
    for path, text in files.items():
        destination = folder / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8", newline="")
    actor = Actor("Offline fixture", "fixture@example.invalid")
    repo.index.add(list(files))
    base = repo.index.commit("base", author=actor, committer=actor)
    (folder / "symbols.py").write_text("FORBIDDEN_FIX_HEAD\n", encoding="utf-8")
    repo.index.add(["symbols.py"])
    head = repo.index.commit("fix", author=actor, committer=actor)
    row = {"repo_owner": "fixture", "repo_name": "repo", "base_sha": base.hexsha,
           "issue_url": "https://example.invalid/issues/1", "issue_title": "crash",
           "issue_body": "Routing a missing alias crashes.", "changed_files": ["symbols.py", "tests/excluded.py"],
           "head_sha": head.hexsha, "patch": "FORBIDDEN_PATCH"}
    repo.close()
    return folder, row


def response(value):
    return ModelResponse(json.dumps(value), usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                         latency_seconds=.001, finish_reason="stop")


class InvestigatingClient:
    """Derive the follow-up identifier/path from the actual preceding result."""
    def __init__(self):
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        if "Tool actions are disabled" in messages[0]["content"]:
            return response({"ranked_files": ["symbols.py"]})
        previous = [json.loads(m["content"])["tool_result"] for m in messages
                    if m["role"] == "user" and "tool_result" in json.loads(m["content"])]
        if not previous:
            action = {"action": "search_code", "query": "crash"}
        elif len(previous) == 1:
            text = "\n".join(s["text"] for s in previous[-1]["snippets"])
            identifier = re.search(r"from symbols import (\w+)", text).group(1)
            action = {"action": "search_code", "query": identifier}
        elif len(previous) == 2:
            snippet = next(s for s in previous[-1]["snippets"] if s["text"].startswith("def "))
            action = {"action": "read_file", "path": snippet["path"],
                      "start_line": snippet["start_line"], "end_line": snippet["end_line"]}
        else:
            action = {"action": "submit_ranking", "paths": [previous[-1]["snippets"][0]["path"]]}
        return response({"decision": action})


class SequenceClient:
    def __init__(self, *values):
        self.values = iter(values)
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        value = next(self.values)
        return value if isinstance(value, ModelResponse) else response(value)
