import hashlib
import os
from typing import Optional

import numpy as np


class DenseEmbedder:
    """Embeds texts with a Hugging Face encoder model into unit-length vectors (cosine similarity = dot product).

    Chunk embeddings are cached on disk, keyed by the model and the exact texts, because every target line of a
    datapoint reuses the same chunks and every run of an arm re-indexes the same repositories.
    """

    def __init__(self, model_name: str, pooling: str = "cls", device: Optional[str] = None, max_length: int = 512,
                 batch_size: int = 32, cache_dir: Optional[str] = None):
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be cls or mean, got {pooling!r}")
        self.model_name, self.pooling = model_name, pooling
        self.max_length, self.batch_size = max_length, batch_size
        self.cache_dir = os.path.expanduser(cache_dir) if cache_dir else None
        self.device = device
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self._model = AutoModel.from_pretrained(self.model_name, dtype=dtype).to(self.device).eval()

    def _encode(self, texts: list[str], truncation_side: str) -> np.ndarray:
        import torch

        self._load()
        self._tokenizer.truncation_side = truncation_side
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths per batch, less padding
        out = np.zeros((len(texts), self._model.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                ids = order[start:start + self.batch_size]
                batch = self._tokenizer([texts[i] for i in ids], padding=True, truncation=True,
                                        max_length=self.max_length, return_tensors="pt").to(self.device)
                hidden = self._model(input_ids=batch["input_ids"],
                                     attention_mask=batch["attention_mask"]).last_hidden_state
                if self.pooling == "mean":
                    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    vectors = (hidden * mask).sum(1) / mask.sum(1)
                else:
                    vectors = hidden[:, 0]
                vectors = torch.nn.functional.normalize(vectors.float(), dim=-1)
                out[ids] = vectors.cpu().numpy()
        return out

    def embed_chunks(self, texts: list[str]) -> np.ndarray:
        """Vectors for chunk texts (right-truncated to `max_length` tokens), cached on disk if a cache dir is set."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        path = None
        if self.cache_dir:
            digest = hashlib.sha1((self.model_name + self.pooling + "\0" + "\0".join(texts)).encode()).hexdigest()
            path = os.path.join(self.cache_dir, f"{digest}.npy")
            if os.path.exists(path):
                return np.load(path)
        vectors = self._encode(texts, truncation_side="right")
        if path:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(path, vectors.astype(np.float16))
            vectors = np.load(path)  # use exactly what a later cached run will see
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        """Vector for a query (left-truncated: the last lines before the target line matter most)."""
        return self._encode([text], truncation_side="left")[0]
