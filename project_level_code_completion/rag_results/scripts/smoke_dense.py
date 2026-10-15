"""Can a code-embedding model load under the pinned transformers, and does it rank sensibly? (CPU only)"""
import sys
import time

import torch
from transformers import AutoModel, AutoTokenizer

name = sys.argv[1]
t0 = time.time()
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = AutoModel.from_pretrained(name, trust_remote_code=True).eval()
print(f"loaded {name} in {time.time() - t0:.1f}s; parameters {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

texts = [
    "def read_json(path):\n    with open(path) as f:\n        return json.load(f)",       # relevant to the query
    "class HttpClient:\n    def get(self, url):\n        return requests.get(url)",     # unrelated
    "def write_csv(rows, path):\n    with open(path, 'w') as f:\n        csv.writer(f).writerows(rows)",
]
query = "data = json.load(open(config_path))\nconfig = parse_config(data)"
with torch.no_grad():
    batch = tok([query] + texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    out = model(**{k: v for k, v in batch.items() if k in ("input_ids", "attention_mask")})
    if isinstance(out, torch.Tensor):
        emb = out
    else:
        hidden = out.last_hidden_state
        if len(sys.argv) > 2 and sys.argv[2] == "mean":
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            emb = (hidden * mask).sum(1) / mask.sum(1)
        else:
            emb = hidden[:, 0]
    emb = torch.nn.functional.normalize(emb.float(), dim=-1)
sims = (emb[0:1] @ emb[1:].T)[0].tolist()
print("embedding shape", tuple(emb.shape), "| cosine to query:", [round(s, 3) for s in sims])
print("ranks the json reader first:", sims.index(max(sims)) == 0)
