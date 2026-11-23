# E4 retrieval failure or reasoning failure (CodeLlama-7B, medium, 224 datapoints, 8,676 lines per arm)

Produced by `python -m eval.failure_split` (log section 8.76). Lexical screen: X(L) = identifiers of the target line that the repository defines in another file and the file's own prefix does not define.
"Names visible" = every name of X(L) occurs as an identifier in the prompt the model saw. Per-line rows (13 MB) are not in git; they are in `~/lca_remote/E4/failure_split.jsonl` and reproducible with the script.

## Every line in one of four classes

| arm | lines | right | wrong, self-contained | wrong, retrieval miss | wrong, reasoning (names visible) |
|---|---:|---:|---:|---:|---:|
| e4_F_N1753 | 8676 | 4523 (52.1%) | 2755 (31.8%) | 171 (2.0%) | 1227 (14.1%) |
| e4_pd_N1753 | 8676 | 3444 (39.7%) | 3139 (36.2%) | 514 (5.9%) | 1579 (18.2%) |
| e4_F_N12000 | 8676 | 4625 (53.3%) | 2721 (31.4%) | 51 (0.6%) | 1279 (14.7%) |
| e4_pd_N12000 | 8676 | 4146 (47.8%) | 2916 (33.6%) | 199 (2.3%) | 1415 (16.3%) |
| e4_F_16k | 8676 | 4625 (53.3%) | 2716 (31.3%) | 45 (0.5%) | 1290 (14.9%) |
| e4_pd_16k | 8676 | 4181 (48.2%) | 2916 (33.6%) | 170 (2.0%) | 1409 (16.2%) |

## Lines that need something from another file (X(L) not empty: 3,303 lines, 38.1 % of all)

| arm | EM, X(L) empty | EM, X(L) not empty | all names visible | EM if all visible | EM if a name is missing | definitions of all names visible |
|---|---:|---:|---:|---:|---:|---:|
| e4_F_N1753 | 0.487 | 0.577 | 94.3 % | 0.606 (3,114 lines) | 0.095 (189) | 15.8 % |
| e4_pd_N1753 | 0.416 | 0.366 | 82.6 % | 0.421 (2,727) | 0.108 (576) | 6.1 % |
| e4_F_N12000 | 0.494 | 0.597 | 98.3 % | 0.606 (3,246) | 0.105 (57) | 37.6 % |
| e4_pd_N12000 | 0.457 | 0.511 | 93.2 % | 0.540 (3,079) | 0.112 (224) | 30.0 % |
| e4_F_16k | 0.495 | 0.596 | 98.5 % | 0.603 (3,253) | 0.100 (50) | 39.3 % |
| e4_pd_16k | 0.457 | 0.522 | 94.2 % | 0.547 (3,110) | 0.119 (193) | 34.0 % |

## Check of the rebuilt prompts

Rebuilt prompt token count (+ the BOS token) equals the logged `input_tokens` for 99.6 / 99.2 / 99.3 / 98.6 % of lines for the arms that are not cropped (F N1753, pd N1753, F N12000, F 16K). For the arms where the generator crops the prompt to
the window (pd N12000: 24.6 % of lines; pd 16K: every line) the logged count is exactly one lower than the rebuilt one, because the generator's cropped input includes the BOS token inside the 15,900-token limit; the rebuilt window
therefore starts one token earlier, which can only matter for the first, partial token.
