```python
import argparse
import os
from typing import List, Tuple
import numpy as np

class DummyErnieTokenizer:
    def __init__(self, vocab_path, use_fast=False):
        self.vocab_path = vocab_path
        self.use_fast = use_fast

    def encode(self, text, max_length, padding='max_length', truncation=True):
        # Dummy encoding: convert chars to ord, pad/truncate to max_length
        tokens = [ord(c) % 100 for c in text][:max_length]
        if len(tokens) < max_length:
            tokens += [0] * (max_length - len(tokens))
        return tokens

class DummyErnieRuntime:
    def __init__(self, model_dir, device, backend, use_fp16):
        self.model_dir = model_dir
        self.device = device
        self.backend = backend
        self.use_fp16 = use_fp16

    def infer(self, batch_inputs):
        # Dummy inference: random logits for 2 classes
        batch_size = len(batch_inputs)
        logits = np.random.randn(batch_size, 2)
        return logits

def batchify(data: List[str], batch_size: int) -> List[List[str]]:
    return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

class ErnieSequenceClassifier:
    def __init__(self, model_dir: str, vocab_path: str, device: str, backend: str,
                 use_fp16: bool, use_fast_tokenizer: bool, max_seq_length: int):
        self.tokenizer = DummyErnieTokenizer(vocab_path, use_fast=use_fast_tokenizer)
        self.runtime = DummyErnieRuntime(model_dir, device, backend, use_fp16)
        self.max_seq_length = max_seq_length
        self.label_map = {0: "negative", 1: "positive"}

    def preprocess(self, texts: List[str]) -> np.ndarray:
        input_ids = [self.tokenizer.encode(text, self.max_seq_length) for text in texts]
        return np.array(input_ids, dtype=np.int64)

    def infer(self, input_ids: np.ndarray) -> np.ndarray:
        logits = self.runtime.infer(input_ids)
        return logits

    def postprocess(self, logits: np.ndarray) -> List[Tuple[str, float]]:
        probs = self.softmax(logits)
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        results = [(self.label_map[pred], float(conf)) for pred, conf in zip(preds, confs)]
        return results

    def predict(self, texts: List[str]) -> List[Tuple[str, float]]:
        input_ids = self.preprocess(texts)
        logits = self.infer(input_ids)
        results = self.postprocess(logits)
        return results

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

def main():
    parser = argparse.ArgumentParser(description="Ernie Sequence Classification Predictor")
    parser.add_argument('--model_dir', type=str, required=True, help='Path to model directory')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to tokenizer vocab file')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'], help='Inference device')
    parser.add_argument('--backend', type=str, default='onnx', choices=['onnx', 'paddle'], help='Runtime backend')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--use_fp16', action='store_true', help='Use FP16 mode')
    parser.add_argument('--use_fast_tokenizer', action='store_true', help='Use fast tokenizer')
    parser.add_argument('--input_file', type=str, required=True, help='Input text file, one sentence per line')
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    predictor = ErnieSequenceClassifier(
        model_dir=args.model_dir,
        vocab_path=args.vocab_path,
        device=args.device,
        backend=args.backend,
        use_fp16=args.use_fp16,
        use_fast_tokenizer=args.use_fast_tokenizer,
        max_seq_length=args.max_seq_length
    )

    batches = batchify(texts, args.batch_size)
    example_id = 0
    for batch_id, batch_texts in enumerate(batches):
        results = predictor.predict(batch_texts)
        for i, (label, score) in enumerate(results):
            print(f"Batch {batch_id}\tExample {example_id}\tInput: {batch_texts[i]}\tPredicted: {label}\tConfidence: {score:.4f}")
            example_id += 1
        if (batch_id + 1) % args.log_interval == 0:
            print(f"Processed {example_id} examples.")

if __name__ == "__main__":
    main()
```