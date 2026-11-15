```python
import argparse
import os
from typing import List, Tuple, Dict, Any

# Placeholder imports for Ernie model, tokenizer, and runtime
# Replace these with actual imports from your deep learning framework
# For example: from paddlenlp.transformers import ErnieTokenizer, ErnieForSequenceClassification
class ErnieTokenizer:
    def __init__(self, vocab_file, use_fast=False):
        pass
    def __call__(self, texts, max_length, padding, truncation, return_tensors):
        # Dummy implementation
        return {"input_ids": [[1,2,3]], "attention_mask": [[1,1,1]]}

class ErnieRuntime:
    def __init__(self, model_dir, device, backend, use_fp16):
        pass
    def infer(self, inputs):
        # Dummy implementation: returns logits
        return [[0.1, 0.9]]

def batchify(data: List[str], batch_size: int) -> List[List[str]]:
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i+batch_size])
    return batches

class ErnieSequenceClassifier:
    def __init__(
        self,
        model_dir: str,
        vocab_path: str,
        device: str = "cpu",
        backend: str = "default",
        use_fp16: bool = False,
        use_fast_tokenizer: bool = False,
        max_seq_length: int = 128
    ):
        self.tokenizer = ErnieTokenizer(vocab_file=vocab_path, use_fast=use_fast_tokenizer)
        self.runtime = ErnieRuntime(model_dir=model_dir, device=device, backend=backend, use_fp16=use_fp16)
        self.max_seq_length = max_seq_length

    def preprocess(self, texts: List[str]) -> Dict[str, Any]:
        return self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )

    def infer(self, inputs: Dict[str, Any]) -> List[List[float]]:
        return self.runtime.infer(inputs)

    def postprocess(self, logits: List[List[float]]) -> Tuple[List[int], List[float]]:
        import numpy as np
        probs = [self._softmax(logit) for logit in logits]
        labels = [int(np.argmax(p)) for p in probs]
        confs = [float(np.max(p)) for p in probs]
        return labels, confs

    def _softmax(self, x):
        import numpy as np
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        inputs = self.preprocess(texts)
        logits = self.infer(inputs)
        labels, confs = self.postprocess(logits)
        results = []
        for i, (text, label, conf) in enumerate(zip(texts, labels, confs)):
            results.append({
                "example_id": i,
                "input": text,
                "predicted_label": label,
                "confidence": conf
            })
        return results

def parse_args():
    parser = argparse.ArgumentParser(description="Ernie Sequence Classification CLI")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to tokenizer vocab file")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Inference device")
    parser.add_argument("--backend", type=str, default="default", help="Runtime backend")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 mode")
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Use fast tokenizer")
    parser.add_argument("--input_file", type=str, required=True, help="Input text file, one sentence per line")
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
        use_fast_tokenizer=args.use_fast_tokenizer,
        max_seq_length=args.max_seq_length
    )
    batches = batchify(texts, args.batch_size)
    example_id_offset = 0
    for batch_id, batch in enumerate(batches):
        results = classifier.predict(batch)
        for result in results:
            print(f"batch_id: {batch_id}\texample_id: {example_id_offset + result['example_id']}\t"
                  f"input: {result['input']}\tpredicted_label: {result['predicted_label']}\t"
                  f"confidence: {result['confidence']:.4f}")
        example_id_offset += len(batch)

if __name__ == "__main__":
    main()
```