```python
import argparse
import os
from typing import List, Tuple, Dict, Any
import numpy as np

# Placeholder imports for Ernie model and tokenizer
from paddlenlp.transformers import ErnieTokenizer
from paddlenlp.runtime import Runtime

def parse_args():
    parser = argparse.ArgumentParser(description="Ernie Sequence Classification Prediction CLI")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of the model.")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the tokenizer vocab file.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Inference device.")
    parser.add_argument("--backend", type=str, default="onnxruntime", choices=["onnxruntime", "paddle"], help="Runtime backend.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval.")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mode.")
    parser.add_argument("--fast_tokenizer", action="store_true", help="Use fast tokenizer.")
    parser.add_argument("--input_file", type=str, required=True, help="Input file with one sentence per line.")
    return parser.parse_args()

def batchify(data: List[Any], batch_size: int) -> List[List[Any]]:
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

class ErnieSequenceClassifier:
    def __init__(
        self,
        model_dir: str,
        vocab_path: str,
        device: str = "cpu",
        backend: str = "onnxruntime",
        max_seq_length: int = 128,
        fp16: bool = False,
        fast_tokenizer: bool = False
    ):
        self.tokenizer = ErnieTokenizer.from_pretrained(vocab_path, use_fast=fast_tokenizer)
        self.runtime = Runtime(
            model_dir=model_dir,
            device=device,
            backend=backend,
            fp16=fp16
        )
        self.max_seq_length = max_seq_length

    def preprocess(self, texts: List[str]) -> Dict[str, np.ndarray]:
        encoded = self.tokenizer(
            texts,
            max_seq_len=self.max_seq_length,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True
        )
        return {
            "input_ids": np.array(encoded["input_ids"], dtype="int64"),
            "token_type_ids": np.array(encoded["token_type_ids"], dtype="int64"),
            "attention_mask": np.array(encoded["attention_mask"], dtype="int64")
        }

    def infer(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        outputs = self.runtime.run(inputs)
        return outputs[0]  # Assuming logits are the first output

    def postprocess(self, logits: np.ndarray) -> Tuple[List[int], List[float]]:
        probs = self.softmax(logits)
        labels = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        return labels.tolist(), confidences.tolist()

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        inputs = self.preprocess(texts)
        logits = self.infer(inputs)
        labels, confidences = self.postprocess(logits)
        results = []
        for i, (text, label, conf) in enumerate(zip(texts, labels, confidences)):
            results.append({
                "example_id": i,
                "sentence": text,
                "label": label,
                "confidence": conf
            })
        return results

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

def main():
    args = parse_args()
    with open(args.input_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    batches = batchify(texts, args.batch_size)
    predictor = ErnieSequenceClassifier(
        model_dir=args.model_dir,
        vocab_path=args.vocab_path,
        device=args.device,
        backend=args.backend,
        max_seq_length=args.max_seq_length,
        fp16=args.fp16,
        fast_tokenizer=args.fast_tokenizer
    )
    example_id_offset = 0
    for batch_id, batch_texts in enumerate(batches):
        results = predictor.predict(batch_texts)
        for result in results:
            print(f"batch_id: {batch_id}\texample_id: {example_id_offset + result['example_id']}\tsentence: {result['sentence']}\tpredicted_label: {result['label']}\tconfidence: {result['confidence']:.4f}")
        example_id_offset += len(batch_texts)

if __name__ == "__main__":
    main()
```