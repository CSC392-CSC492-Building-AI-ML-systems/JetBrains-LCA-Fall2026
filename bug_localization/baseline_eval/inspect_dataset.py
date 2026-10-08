from datasets import load_dataset

print("Loading the dataset...")

dataset = load_dataset(
    "JetBrains-Research/lca-bug-localization",
    "py",
    split="test",
)

example = dataset[0]

print("\nNumber of examples:", len(dataset))
print("\nTitle:", example["issue_title"])
print("\nDescription:", example["issue_body"])
print("\nRepository:", example["repo_owner"], example["repo_name"])
print("\nBuggy commit:", example["base_sha"])
print("\nCorrect files:", example["changed_files"])