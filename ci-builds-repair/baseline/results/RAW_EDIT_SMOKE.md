# Raw-code edit protocol and model comparison

The matched synthetic comparison uses seven handwritten fixtures, including
one multiline block, and three seeds. It keeps the prompt, grammar, parser,
sampling temperature, 192-token cap, and scorer fixed between the two
constrained models. Each checkpoint uses its native chat template and
tokenizer. The scorer compares the resulting Python AST with the expected
AST without executing the code.

| Model and output protocol | Format-valid | Python parses | Expected AST matches |
| --- | ---: | ---: | ---: |
| Mistral, raw actions without constraint | 0/21 | 0/21 | 0/21 |
| Mistral, constrained raw actions | 21/21 | 14/21 | 6/21 |
| Qwen Coder, constrained raw actions | 21/21 | 21/21 | 15/21 |

I verified and re-scored all 42 Mistral responses and all 21 Qwen responses.
The Qwen model is the [4-bit MLX conversion](https://huggingface.co/mlx-community/Qwen2.5-Coder-7B-Instruct-4bit/tree/019cc73c45c770444708a6dd8690c66243cc5c80)
of [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct),
pinned to revision `019cc73c45c770444708a6dd8690c66243cc5c80`. Local model
checksums and package versions are recorded in the comparison manifest.

| Fixture | Mistral expected AST matches | Qwen expected AST matches |
| --- | ---: | ---: |
| Indented arithmetic | 0/3 | 0/3 |
| Two required edits | 3/3 | 3/3 |
| One edit, one omission | 0/3 | 3/3 |
| Delete a block | 0/3 | 3/3 |
| Preserve quoted text | 0/3 | 0/3 |
| No edits needed | 3/3 | 3/3 |
| Multiline replacement | 0/3 | 3/3 |

Mistral now handled the no-op correctly, but deleted the only function
body in the arithmetic trials and still failed the multiline case. Qwen
repaired the multiline block and got deletion and omission right. It still
replaced the arithmetic line with an entire nested function definition
instead of a line replacement, and left the quoted-text bug unchanged.
The 21 trials represent only seven independent fixture designs.

**Decision:** Qwen with raw edits passes the existing synthetic gate;
Mistral does not. This is enough to evaluate Qwen in a separate, matched
no-retrieval development experiment on real CI cases. Keep the paper-style
Mistral baseline unchanged and use the same model in both arms of any future
retrieval comparison. Measure patch application, syntax, and reference
agreement before comparing retrieval. Pay particular attention to nested
definitions and formatting-only fixes: restoring old base indentation is
not suitable for every kind of repair. No benchmark models, build checks,
retrieval, or GitHub Actions ran in these smoke tests.

Run signatures: Mistral `2d9f6819a99991705adf`; matched model comparison
`6eefc4788c3efc1fd144`. Raw predictions, manifests, and metrics are stored
under the ignored benchmark output directories.
