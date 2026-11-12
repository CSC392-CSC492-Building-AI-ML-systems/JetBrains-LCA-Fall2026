"""Export compact, auditable offline benchmark results for version control."""

import argparse
import json
import statistics
from pathlib import Path

from run_benchmark import save_json, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    manifest = json.loads((args.run / "run_metadata.json").read_text())
    metrics = summarize(args.run, manifest)
    save_json(args.output / "metrics.json", metrics)
    save_json(args.output / "run_metadata.json", manifest)
    lines = ["# Local baseline versus BM25 RAG benchmark", "",
             f"Run signature: `{args.run.name}`. Dataset: {manifest['dataset']}, "
             f"{len(manifest['ids'])}/{manifest['dataset_size']} selected records. "
             f"Seeds: {manifest['seeds']}. Completed predictions: "
             f"{metrics['evaluated_predictions']}/{metrics['expected_predictions']}.", "",
             "This is an offline reference-patch and syntax benchmark. No project code, "
             "GitHub workflow, or GitHub Actions evaluator was executed. These numbers are not CI Pass@1.", "",
             f"Model: `{manifest['model_id']}`; temperature {manifest['temperature']}; "
             f"context {manifest['context_window']} tokens; output cap {manifest['max_tokens']} tokens. "
             "Both methods use the same oracle edit locations and effective seed = seed + dataset ID. "
             "RAG adds BM25 context from the failing snapshot. Settings were fixed before generation.", "",
             "| Method | Seed | Evaluated | Applied response | Exact reference | Edit-token F1 | Nonempty | Syntax preserved among nonempty |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    def fmt(value):
        return "n/a" if value is None else f"{value:.4f}"
    for row in metrics["per_seed"]:
        lines.append(f"| {row['method']} | {row['seed']} | {row['evaluated']}/{row['selected']} | "
                     f"{fmt(row['valid_response_rate'])} | {fmt(row['patch_exact_match'])} | "
                     f"{fmt(row['edit_token_f1'])} | {fmt(row['nonempty_patch'])} | "
                     f"{fmt(row['syntax_preserved_nonempty'])} (n={row['syntax_nonempty_denominator']}) |")
    lines += ["", "## Across seeds", "", "| Method | Exact reference, mean ± std | Edit-token F1, mean ± std |",
              "| --- | --- | --- |"]
    for method, values in metrics["across_seeds"].items():
        exact, f1 = values["patch_exact_match"], values["edit_token_f1"]
        lines.append(f"| {method} | {fmt(exact['mean'])} ± {fmt(exact['std_across_seeds'])} | "
                     f"{fmt(f1['mean'])} ± {fmt(f1['std_across_seeds'])} |")
    rag = [row for row in metrics["per_seed"] if row["method"] == "rag" and row["evaluated"]]
    if rag:
        lines += ["", f"RAG reference-changed-file retrieval recall: "
                  f"{fmt(rag[0]['retrieval_file_recall_before_packing'])} before packing; "
                  f"{fmt(rag[0]['retrieval_file_recall_after_packing'])} after packing. "
                  "Targets already supplied as editable blocks are excluded from extra evidence."]
    if all(metrics["across_seeds"].get(method, {}).get("edit_token_f1", {}).get("mean") is not None
           for method in ("baseline", "rag")):
        delta = metrics["across_seeds"]["rag"]["edit_token_f1"]["mean"] - metrics["across_seeds"]["baseline"]["edit_token_f1"]["mean"]
        lines += ["", f"Mean edit-token F1 difference (RAG minus baseline): **{delta:+.4f}**. "
                  "This describes reference-edit agreement, not semantic improvement or statistical significance."]
    lines += ["", "## Interpretation and limitations", "",
              "Exact reference match requires the resulting target-file text to equal the dataset repair; "
              "valid alternative fixes can score zero. Token F1 compares only added/removed tokens, keyed "
              "by file and edit direction, so unchanged context earns no credit. Applied response means "
              "the replacement format validated. Syntax preservation uses Python 3.13 AST parsing only; "
              "it does not execute imports, linters, dependency setup, tests, or platform-specific builds.", "",
              "Malformed/truncated/over-budget model outputs remain in denominators as unapplied, "
              "unchanged predictions. Pipeline/source failures are listed separately and reduce coverage. "
              "Population standard deviations describe variation across the configured seeds.", "",
              "The historical paper used 77 items and did not publish CI repair seeds/decoding settings. "
              "This cached dataset, local 4-bit model, reconstructed prompt, and proxy metrics cannot "
              "establish reproduction of the paper's build-success percentage.", "",
              f"Source/pipeline failure records: {len(metrics['source_failures'])}. "
              f"All selected predictions evaluated: {metrics['complete']}.", "",
              "See `metrics.json` for per-seed denominators, error reasons, syntax regressions, token "
              "counts, and latency. `run_metadata.json` records dataset/configuration checksums and source "
              "hashes. Raw prompts, responses, patch files, and prediction checksums remain in the ignored "
              f"benchmark output under `out/rag-local/{args.run.name}/`.", ""]
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output / "REPORT.md")


if __name__ == "__main__":
    main()
