# Proposal: Retrieval-Augmented Generation for Module Summarization

## Purpose and scope

This proposal describes how retrieval-augmented generation (RAG) could improve the Module Summarization benchmark in Long Code Arena. It first outlines the role of retrieval across the six benchmarks, then proposes a specific retrieval pipeline, expected benefits, limitations, and evaluation method for module summarization. Implementation and experiments are future work; training is deferred.

## How RAG applies to the six benchmarks

RAG selects relevant evidence from a larger code collection and supplies it to a model when it performs a task. The retrieval query and useful evidence depend on the benchmark. The table describes proposed applications, rather than claiming that all existing Long Code Arena baselines implement RAG.

| Benchmark | Query and retrieved evidence | How the evidence supports the task |
|---|---|---|
| Library-based Code Generation | Search with the task description for relevant library functions, classes, signatures, and permitted usage examples. | Helps the model use the library's actual APIs when generating a program. |
| CI Build Repair | Search with the failure log and build details for related configuration, dependencies, and source code. | Gives the model context for diagnosing the failure and proposing a repair. |
| Project-level Code Completion | Search with the incomplete code and imports for available definitions, related functions, and existing usages. | Helps the model complete code consistently with the project. |
| Commit Message Generation | Search with the diff and changed symbols for surrounding implementation context. | Helps explain the purpose and implications of a change. The diff remains the central evidence. |
| Bug Localization | Search with the issue description for source files containing relevant symbols or behavior. | Produces candidate locations for ranking or further model analysis. Retrieval alone can perform localization; generation is optional. |
| Module Summarization | Search with the documentation intent and target filename for relevant implementation chunks. | Gives the model evidence for writing a documentation page about the requested topic. |

These applications follow the task definitions in the [Long Code Arena benchmark suite](https://github.com/JetBrains-Research/lca-baselines). Retrieval must respect each task's permitted snapshot and exclude reference answers or future information.

## Module Summarization: task and current baseline

Module Summarization requires generating documentation from source code and a one-sentence intent describing the expected content. The target documentation filename is also provided. The dataset has 216 test examples from 43 Python repositories and includes the reference documentation and supplied relevant code context. [Dataset card](https://huggingface.co/datasets/JetBrains-Research/lca-module-summarization), [LCA paper, Section 2.6](https://arxiv.org/html/2406.11612v1#S2.SS6).

Our baseline runner uses the dataset's `relevant_code_context`, truncates it to the first 2,000 tokens, and asks the generation model to write the documentation. This provides a simple comparison point, but useful implementation details may occur beyond the retained prefix. For example, code explaining the requested behavior may appear after imports, setup code, or unrelated functions.

The local artifact audit confirmed 216 baseline prediction files and an earlier five-example keyword-retrieval trial. Official CompScore evaluation is still outstanding. These artifacts establish prior work to build on; they do not establish that the proposed approach improves the official benchmark score.

## Research question and hypothesis

**Research question:** With the generation model and a 2,000-token code budget held constant, does intent-guided code retrieval produce better module documentation than prefix truncation?

**Hypothesis:** Retrieval will improve documentation when relevant behavior is located beyond the beginning of the code context. Improvements may be smaller, or negative, for broad overview pages requiring evidence from many parts of a module or for intents that give little guidance about what to retrieve.

## Proposed approach

The initial approach uses BM25, a lexical ranking method that accounts for term frequency, rarity, and document length. It offers an interpretable starting point beyond the earlier prototype's raw keyword-overlap ranking. [BM25 reference](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html).

**Pipeline:** documentation intent and filename → retrieve code chunks → select evidence within 2,000 tokens → generate documentation using the baseline prompt.

1. **Define the retrieval corpus.** The first experiment would retrieve within each example's supplied `relevant_code_context`. This keeps the code pool identical to the baseline and isolates the effect of chunk selection. A later extension would search source files across the benchmark repository snapshot. These are distinct experiments: the first does not test repository-wide file discovery.

2. **Split the code into chunks.** For the initial concatenated context, a proposed starting configuration is approximately 400-token windows with 50-token overlap. Offsets would preserve traceability to the input. In the later repository-wide version, function/class boundaries and source paths would provide more meaningful chunks. These settings are starting choices to validate, not proven optimal values.

3. **Build the query.** Combine the supplied intent with the filename stem, such as `custom_attribute` from `custom_attribute.rst`. Split snake_case and camelCase identifiers while retaining their original forms. The reference documentation must not be used to construct the query.

4. **Retrieve and assemble context.** Rank chunks with BM25, remove duplicate overlapping evidence, and pack useful chunks under the same 2,000-token budget as the baseline. Include separators and metadata in that budget. Record retrieval scores, chunk offsets, selected tokens, and any fallback used when no terms match.

5. **Generate documentation.** Use the existing generation prompt, model, and generation settings. Changing only context selection makes the comparison easier to interpret.

6. **Preserve experiment evidence.** Save the selected context alongside each prediction, together with the example ID, model, tokenizer, retrieval settings, dataset revision, and timing. This would support reproduction and distinguish retrieval failures from generation failures.

The later repository-wide extension should use benchmark snapshots rather than current GitHub branches and exclude the target documentation and copies of it. The supplied relevant-file annotations are derived from reference-document links; using them to choose files would weaken a claim of independent repository-wide discovery. The dataset permits collecting custom context. [Dataset structure and context guidance](https://huggingface.co/datasets/JetBrains-Research/lca-module-summarization#datapoint-structure).

## Expected benefits and limitations

The main expected benefit is better use of a limited context window: the model can receive implementation details relevant to the requested page even when they occur late in the input. Saved retrieval evidence would also make failures easier to investigate.

RAG does not guarantee better documentation. Lexical search can miss semantic matches, generic intents can produce weak queries, and chunking can separate related behavior. A narrow selection may omit the broader relationships needed for an overview. These limitations motivate inspecting retrieval evidence and comparing narrow documentation tasks with broader guides.

Lower cost is also not guaranteed against the existing 2k-token baseline, because both methods have the same context allowance and retrieval adds work. Potential savings against a larger-context approach would require a separate comparison.

Repository-level research such as RepoCoder supports investigating retrieval for code tasks, but its results concern code completion and do not establish gains for this documentation benchmark. [RepoCoder](https://aclanthology.org/2023.emnlp-main.151/).

## Proposed evaluation and future work

The primary comparison would be prefix truncation versus BM25 retrieval over the same supplied code pool. Both conditions would use identical example IDs, model, prompt, tokenizer, 2k code budget, output limit, and generation settings. A small execution check would precede a diverse pilot spanning repositories, documentation types, context sizes, and intent specificity.

The primary quality metric would be official **CompScore**, which compares generated and reference documentation using an LLM assessor with the documentation order swapped. The judge and its code context must remain identical across conditions. Diagnostic ROUGE-L can support inspection but should not replace CompScore. [Official metric](https://huggingface.co/datasets/JetBrains-Research/lca-module-summarization#metric).

Analysis would examine API correctness, unsupported claims, coverage, useful examples, and documentation structure. Retrieval time, generation time, token usage, and judging resources would be reported separately. Failure analysis would distinguish missing evidence from cases where adequate evidence was retrieved but the model still produced poor documentation.

Because the dataset provides a test split, any examples used to tune the method should be explicitly identified. Final evaluation should separate untouched repositories from development examples and report paired differences rather than relying only on aggregate means.

Future implementation would proceed through retrieval-only inspection, integration with the baseline runner, a diverse pilot, and official scoring. Repository-wide retrieval and semantic or hybrid retrieval could follow if justified by the observed failures. This week's deliverable is the written design; no new implementation, training, or experimental results are claimed.
