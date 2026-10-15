import json
import os

from scipy.stats import spearmanr

home = os.path.expanduser("~")
M = f"{home}/lca_artifacts/python/starcoderbase-1b/medium_context/arms"


def rows(arm):
    return [r for r in map(json.loads, open(f"{M}/{arm}/generation_results.jsonl")) if not r["zero_context"]]


a, b = rows("g_path_distance"), rows("s1_function_class_mask_half")
diff = [i for i, (x, y) in enumerate(zip(a, b)) if x["prediction_line"] != y["prediction_line"]]
print(f"path_distance vs function_class_mask_half on medium dev: {len(a)} lines, differing predictions: {len(diff)}")
for i in diff[:3]:
    print("  ", repr(a[i]["prediction_line"][:60]), "|", repr(b[i]["prediction_line"][:60]))

# Rank agreement with the paper's Table 10 perplexity ordering (CodeLlama-7B, medium, 12,000 tokens): lower perplexity = better.
paper = {"path_distance": 1.607, "half_memory_path_distance": 1.636, "function_class_mask_one": 1.710,
         "imports_first": 1.732, "file_length": 1.742, "naive": 1.761, "zero": 1.849}
em = {"path_distance": 0.390, "half_memory_path_distance": 0.359, "function_class_mask_one": 0.350,
      "imports_first": 0.336, "file_length": 0.329, "naive": 0.326, "zero": 0.323}
names = list(paper)
rho, p = spearmanr([-paper[n] for n in names], [em[n] for n in names])
print(f"Spearman between the paper's perplexity ranking and our EM ranking over {len(names)} shared composers: "
      f"{rho:.2f} (p={p:.3g})")
print("paper order:", sorted(names, key=lambda n: paper[n]))
print("our order  :", sorted(names, key=lambda n: -em[n]))
