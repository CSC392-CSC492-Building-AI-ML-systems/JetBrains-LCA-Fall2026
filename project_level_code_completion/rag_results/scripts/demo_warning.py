import warnings

from rag.chunker import chunk_repo

files = {"a.py": "def f():\n    return 1\n", "b.py": "def g():\n    return 2\n", "c.md": "notes"}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    chunks = chunk_repo(files, lambda t: len(t.split()))
print("chunk kinds:", sorted({c.kind for c in chunks}))
for w in caught:
    print(f"{w.category.__name__}: {w.message}")
print("warnings raised:", len(caught))
