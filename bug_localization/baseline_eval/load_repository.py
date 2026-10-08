import json
from pathlib import Path

from datasets import load_dataset
from git import Repo


# Repository clones and outputs live in the parent bug_localization folder.
project_dir = Path(__file__).resolve().parents[1]
repo_dir = project_dir / "repos" / "thealgorithms__python"

# Load the same first example you inspected earlier.
dataset = load_dataset(
    "JetBrains-Research/lca-bug-localization",
    "py",
    split="test",
)
example = dataset[0]

# Confirm the example matches the repository we downloaded.
repo_name = f"{example['repo_owner']}__{example['repo_name']}"
if repo_name.lower() != repo_dir.name.lower():
    raise ValueError(f"This example needs a different repository: {repo_name}")

# Access the repository at the commit where the bug existed.
repo = Repo(repo_dir)
commit = repo.commit(example["base_sha"])

print("Repository:", repo_name)
print("Reading commit:", commit.hexsha)

repo_content = {}

for item in commit.tree.traverse():
    # Only read Python files.
    if item.type != "blob" or not item.path.endswith(".py"):
        continue

    # Match the original baseline's test-directory filter.
    if any(part in item.path.lower() for part in ("test/", "tests/")):
        continue

    # Read the committed file directly from Git history.
    raw_content = item.data_stream.read()

    try:
        repo_content[item.path] = raw_content.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"File needs an explicit encoding: {item.path}")

print("\nPython files loaded:", len(repo_content))
print("\nFirst five file paths:")

for file_path in list(repo_content)[:5]:
    print(" -", file_path)

# Save the snapshot so the next step can reuse it.
output_dir = project_dir / "outputs"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "first_repo_content.json"

snapshot = {
    "repository": repo_name,
    "base_sha": commit.hexsha,
    "files": repo_content,
}

output_path.write_text(
    json.dumps(snapshot, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\nSaved snapshot to:", output_path)
