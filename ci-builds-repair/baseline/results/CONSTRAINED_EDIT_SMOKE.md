# Enforced short-ID edits: synthetic smoke result

Run `1c5e1e46c44b3f21eac9` completed and was independently re-scored from
all 36 checksummed responses. Six handwritten fixtures ran at three seeds
in each arm. The same cached 4-bit Mistral model, prompts, temperature,
192-token limit, and scorer were used throughout. The only generation
difference was the [LLGuidance](https://github.com/guidance-ai/llguidance)
token mask and its stop at the first complete object. No retrieval, project
tests, generated code execution, or GitHub Actions were used.

| Arm | Format-valid | Format-valid and valid Python syntax | Expected AST match |
| --- | ---: | ---: | ---: |
| No constraint | 1/18 | 0/18 | 0/18 |
| Enforced short-ID JSON | 18/18 | 8/18 | 3/18 |

The unconstrained arm had nine length caps, seven invalid JSON answers and
one invalid schema. The constrained arm had no format failures or length
caps, but all 18 answers proposed a change, including all three no-op cases.
Its only expected-AST matches were the three seeds of the two-required-block
fixture. The syntax check and AST comparison did not execute the patches.

| Synthetic fixture | Constrained Python parses | Constrained expected AST matches |
| --- | ---: | ---: |
| Indented arithmetic edit | 0/3 | 0/3 |
| Two required edits | 3/3 | 3/3 |
| One edit, one omission | 0/3 | 0/3 |
| Delete a block | 3/3 | 0/3 |
| Preserve quotes in a string | 0/3 | 0/3 |
| Leave all code unchanged | 2/3 | 0/3 |

The grammar constrained the edit envelope, not its contents. The model
omitted function-body indentation, emitted JSON-like text in place of
Python assignments, and made unsolicited edits on the no-op case. This
explains why all envelopes were accepted yet only three edits matched.

**Decision:** The synthetic gate failed. Do not advance to the CI-repair
development benchmark or RAG yet. The next useful ablation is to separate
the constrained keep/delete/replace decisions from the raw replacement
text, with explicit indentation handling, and re-test the same fixtures
plus a multi-line block. Keep envelope validity, syntax and reference
agreement separate. If the model still cannot reliably produce the toy
edits, compare a stronger code model on the same fixtures before spending
more time on retrieval. Neither synthetic score is a build-success estimate.
