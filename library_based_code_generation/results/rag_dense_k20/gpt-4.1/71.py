```python
import argparse
import os
from typing import List, Tuple, Dict, Any

# Placeholder imports for Ernie model, tokenizer, and runtime
# Replace these with actual imports from your library
# from your_library import ErnieTokenizer, ErnieRuntime, batch_predict

class DummyTokenizer:
    def __init__(self, vocab_path, use_fast):
        self.vocab_path = vocab_path
        self.use_fast = use_fast

    def encode(self, text, max_length):
        # Dummy encoding: returns list of token ids (simulate)
        return [ord(c) % 100 for c in text][:max_length]

    def batch_encode(self, texts, max_length):
        return [self.encode(t, max_length) for t in texts]

class DummyRuntime:
    def __init__(self, model_dir, device, backend, use_fp16):
        self.model_dir = model_dir
        self.device = device
        self.backend = backend
        self.use_fp16 = use_fp16

    def predict(self, batch_inputs):
        # Dummy prediction: returns random logits
        import numpy as np
        batch_size = len(batch_inputs)
        num_labels = 2
        logits = np.random.rand(batch_size, num_labels)
        return logits

def batchify(data: List[str], batch_size: int) -> List[List[str]]:
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i+batch_size])
    return batches

class ErnieSequenceClassifier:
    def __init__(self, model_dir: str, vocab_path: str, device: str, backend: str,
                 use_fp16: bool, use_fast: bool, max_seq_length: int):
        # self.tokenizer = ErnieTokenizer(vocab_path, use_fast=use_fast)
        # self.runtime = ErnieRuntime(model_dir, device=device, backend=backend, use_fp16=use_fp16)
        self.tokenizer = DummyTokenizer(vocab_path, use_fast)
        self.runtime = DummyRuntime(model_dir, device, backend, use_fp16)
        self.max_seq_length = max_seq_length
        self.label_map = {0: "negative", 1: "positive"}

    def preprocess(self, texts: List[str]) -> List[List[int]]:
        return self.tokenizer.batch_encode(texts, self.max_seq_length)

    def inference(self, batch_inputs: List[List[int]]) -> Any:
        return self.runtime.predict(batch_inputs)

    def postprocess(self, logits) -> List[Tuple[int, float]]:
        import numpy as np
        probs = self.softmax(logits)
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)
        return list(zip(preds.tolist(), confs.tolist()))

    def softmax(self, x):
        import numpy as np
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        batch_inputs = self.preprocess(texts)
        logits = self.inference(batch_inputs)
        results = self.postprocess(logits)
        outputs = []
        for idx, (pred, conf) in enumerate(results):
            outputs.append({
                "example_id": idx,
                "input": texts[idx],
                "predicted_label": self.label_map.get(pred, str(pred)),
                "confidence": conf
            })
        return outputs

def parse_args():
    parser = argparse.ArgumentParser(description="Ernie Sequence Classification Prediction CLI")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to tokenizer vocab file")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Inference device")
    parser.add_argument("--backend", type=str, default="onnxruntime", help="Runtime backend")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 mode")
    parser.add_argument("--use_fast", action="store_true", help="Use fast tokenizer")
    parser.add_argument("--input_file", type=str, required=True, help="Input text file (one sentence per line)")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.input_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    classifier = ErnieSequenceClassifier(
        model_dir=args.model_dir,
        vocab_path=args.vocab_path,
        device=args.device,
        backend=args.backend,
        use_fp16=args.use_fp16,
        use_fast=args.use_fast,
        max_seq_length=args.max_seq_length
    )
    batches = batchify(texts, args.batch_size)
    example_id_offset = 0
    for batch_id, batch_texts in enumerate(batches):
        results = classifier.predict(batch_texts)
        for res in results:
            print(f"batch_id: {batch_id}\texample_id: {example_id_offset + res['example_id']}\t"
                  f"input: {res['input']}\tpredicted_label: {res['predicted_label']}\t"
                  f"confidence: {res['confidence']:.4f}")
        example_id_offset += len(batch_texts)

if __name__ == "__main__":
    main()
```