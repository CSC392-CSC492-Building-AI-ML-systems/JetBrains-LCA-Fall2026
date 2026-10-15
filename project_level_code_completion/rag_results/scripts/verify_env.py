import torch, transformers, datasets, hydra

print("torch", torch.__version__, "cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.randn(256, 256, device="cuda")
print("cuda matmul ok:", bool((x @ x).sum().item() != 0))
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("hydra", hydra.__version__)

from tree_sitter_parser.parsed_file import ParsedFile

pf = ParsedFile(code="def f(x):\n    return x\n")
print("ParsedFile ok:", type(pf).__name__)
