"""Static vocabulary: the six Long Code Arena tasks, context strategies and the
failure taxonomy from the project proposal (section 7.2)."""

from __future__ import annotations

from .models import Option, TaskInfo

TASKS: list[TaskInfo] = [
    TaskInfo(
        id="library_based_code_generation",
        label="Library-based Code Generation",
        metric="API Recall",
        higher_is_better=True,
    ),
    TaskInfo(
        id="project_level_code_completion",
        label="Project-level Code Completion",
        metric="Exact Match",
        higher_is_better=True,
    ),
    TaskInfo(id="bug_localization", label="Bug Localization", metric="MAP", higher_is_better=True),
    TaskInfo(id="ci_builds_repair", label="CI Build Repair", metric="Pass rate", higher_is_better=True),
    TaskInfo(
        id="commit_message_generation",
        label="Commit Message Generation",
        metric="ChrF",
        higher_is_better=True,
    ),
    TaskInfo(
        id="module_summarization",
        label="Module Summarization",
        metric="CompScore",
        higher_is_better=True,
    ),
]

APPROACHES: list[Option] = [
    Option(id="baseline", label="Baseline", description="Official Long Code Arena baseline configuration"),
    Option(id="rag", label="RAG", description="Repository retrieval: chunk, embed and retrieve top-k context"),
    Option(id="long_context", label="Long context", description="Supply a large slice of the repository directly"),
    Option(id="agent", label="Agent", description="Multi-step repository search and reasoning"),
]

FAILURE_CATEGORIES: list[Option] = [
    Option(
        id="missing_context",
        label="Missing context",
        description="The file holding the needed information was never retrieved or supplied.",
    ),
    Option(
        id="irrelevant_context",
        label="Irrelevant context",
        description="Supplied context crowded out the useful parts.",
    ),
    Option(
        id="cross_file",
        label="Cross-file reasoning",
        description="Relevant files were present but the model failed to connect them.",
    ),
    Option(
        id="reasoning",
        label="Reasoning despite context",
        description="Correct context was available, yet the answer was wrong.",
    ),
    Option(id="incorrect_code", label="Incorrect code", description="Generated code is invalid or does not run."),
    Option(
        id="context_limit",
        label="Context / compute limit",
        description="Truncation, timeouts or budget limits cut the attempt short.",
    ),
    Option(
        id="uncategorized",
        label="Not yet analysed",
        description="A failed instance that has not been given a root cause yet.",
    ),
]

TASK_IDS = {t.id for t in TASKS}
APPROACH_IDS = {a.id for a in APPROACHES}
CATEGORY_IDS = {c.id for c in FAILURE_CATEGORIES}

DEFAULT_STAGES = ["Load repository", "Retrieve context", "Generate prediction", "Score output"]
STAGES_BY_APPROACH = {
    "baseline": ["Load repository", "Build prompt", "Generate prediction", "Score output"],
    "rag": DEFAULT_STAGES,
    "long_context": ["Load repository", "Pack context window", "Generate prediction", "Score output"],
    "agent": ["Load repository", "Search repository", "Generate prediction", "Score output"],
}


def task_label(task_id: str) -> str:
    return next((t.label for t in TASKS if t.id == task_id), task_id.replace("_", " ").title())


def approach_label(approach_id: str) -> str:
    return next((a.label for a in APPROACHES if a.id == approach_id), approach_id.replace("_", " ").title())
