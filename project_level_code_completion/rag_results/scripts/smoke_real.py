import statistics
import sys
import time

from datasets import load_dataset
from transformers import AutoTokenizer

from data_classes.datapoint_commit_dataset import DatapointCommitDataset
from rag.builder import RagContextBuilder

tok = AutoTokenizer.from_pretrained(sys.argv[1])
count_tokens = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])

ds = load_dataset("JetBrains-Research/lca-project-level-code-completion", "small_context", split="test")
B = 8000 - 100
worst = 0
for hf_dp in ds.select(range(0, 144, 24)):
    names, contents = hf_dp["repo_snapshot"]["filename"], hf_dp["repo_snapshot"]["content"]
    dp = DatapointCommitDataset(
        repo_id=0, repo_name=hf_dp["repo"], completion_lines=hf_dp["completion_lines"],
        context_dict=dict(zip(names, contents)),
        completion_dict={hf_dp["completion_file"]["filename"]: hf_dp["completion_file"]["content"]},
    )
    path = hf_dp["completion_file"]["filename"]
    builder = RagContextBuilder(dp.get_context(), path, dp.repo_name, count_tokens)
    lines = [l for ls in dp.completion_lines.values() for l in ls][:20]
    times, used, n_sel = [], [], []
    for ln in lines:
        t1 = time.time()
        r = builder.context_for(dp.get_prefix(ln), budget=B)
        times.append(time.time() - t1)
        used.append(count_tokens(r.context + "\n" + r.prefix))
        n_sel.append(len(r.chunks))
        assert all(c.path != path for c in r.chunks)
    worst = max(worst, max(used))
    print(f"{dp.repo_name[:40]:40s} chunks={len(builder.chunks):5d} lines={len(lines):2d} "
          f"ctx_ms={statistics.mean(times)*1000:4.0f} tokens max={max(used)} chunks_sel={statistics.mean(n_sel):.1f}")
print(f"worst-case tokens used: {worst} (B={B}) -> {'OK' if worst <= B else 'OVER BUDGET'}")
