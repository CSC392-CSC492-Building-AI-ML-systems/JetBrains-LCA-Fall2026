# Agent development artifacts

The initial config selected tasks 13, 91, 102, 160, and 212. Task 13 completed with the original runner. On task 91, the model started drafting during a search turn and hit its output limit. The recovery reused four saved turns and made the final writing request after the runner fix. The recovered output records the fix's source hash. Tasks 102, 160, and 212 ran with the revised runner under the remainder config.

These artifacts record development behavior. The official comparison uses the full run in the parent arm's results directory.
