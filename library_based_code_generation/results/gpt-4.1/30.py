```python
import os
import torch
import thinc
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Iterator
from transformers import (
    AutoTokenizer,
    AutoModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from thinc.api import Model, chain, with_getitem, list2ragged, Ragged, Softmax, Adam, Optimizer
from thinc.types import Floats2d, Ints2d

# ------------------- Dataclass for Encoded Batch -------------------

@dataclass
class EncodedBatch:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    labels: np.ndarray

# ------------------- Configuration -------------------

CONFIG = {
    "model_name": "distilbert-base-uncased",
    "num_labels": 9,
    "learning_rate": 5e-5,
    "epochs": 3,
    "batch_size": 16,
    "max_length": 128,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "train_data_path": "train.txt",
    "eval_data_path": "eval.txt",
}

# ------------------- Tokenizer -------------------

def create_transformer_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)

# ------------------- Transformer Model -------------------

def create_transformer_model(model_name: str):
    return AutoModel.from_pretrained(model_name)

# ------------------- Transformer Tagger Model -------------------

def create_transformer_tagger_model(transformer, num_labels: int) -> Model:
    def forward(model: Model, X: Tuple[np.ndarray, np.ndarray], is_train: bool):
        input_ids, attention_mask = X
        input_ids_torch = torch.tensor(input_ids).to(CONFIG["device"])
        attention_mask_torch = torch.tensor(attention_mask).to(CONFIG["device"])
        outputs = transformer(
            input_ids=input_ids_torch,
            attention_mask=attention_mask_torch,
        )
        last_hidden = outputs.last_hidden_state
        logits = model.get_param("W")(last_hidden)
        logits_np = logits.detach().cpu().numpy()
        return logits_np, lambda d_logits: None  # No backward for simplicity

    def init(model: Model, X=None, Y=None):
        model.set_param("W", torch.nn.Linear(transformer.config.hidden_size, CONFIG["num_labels"]).to(CONFIG["device"]))

    return Model(
        "transformer_tagger",
        forward,
        init=init,
        dims={"nO": num_labels},
    )

# ------------------- Data Loading -------------------

def load_dataset(path: str, tokenizer, label2id: Dict[str, int], max_length: int) -> List[EncodedBatch]:
    sentences, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        sentence, label = [], []
        for line in f:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence, label = [], []
            else:
                splits = line.strip().split()
                sentence.append(splits[0])
                label.append(splits[-1])
        if sentence:
            sentences.append(sentence)
            labels.append(label)
    encoded_batches = []
    for sent, lab in zip(sentences, labels):
        encoding = tokenizer.batch_encode_plus(
            [sent],
            is_split_into_words=True,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        label_ids = np.ones((1, max_length), dtype=np.int64) * -100
        word_ids = encoding.word_ids(batch_index=0)
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx < len(lab):
                label_ids[0, i] = label2id.get(lab[word_idx], 0)
        encoded_batches.append(
            EncodedBatch(
                input_ids=encoding["input_ids"][0],
                attention_mask=encoding["attention_mask"][0],
                labels=label_ids[0],
            )
        )
    return encoded_batches

# ------------------- Input/Output Conversion -------------------

def convert_inputs(batch: List[EncodedBatch]) -> Tuple[np.ndarray, np.ndarray]:
    input_ids = np.stack([b.input_ids for b in batch])
    attention_mask = np.stack([b.attention_mask for b in batch])
    return input_ids, attention_mask

def convert_outputs(batch: List[EncodedBatch]) -> np.ndarray:
    return np.stack([b.labels for b in batch])

# ------------------- Evaluation -------------------

def evaluate_sequences(preds: np.ndarray, labels: np.ndarray, ignore_index: int = -100) -> float:
    preds_flat = preds.argmax(-1).flatten()
    labels_flat = labels.flatten()
    mask = labels_flat != ignore_index
    correct = (preds_flat[mask] == labels_flat[mask]).sum()
    total = mask.sum()
    return float(correct) / float(total) if total > 0 else 0.0

# ------------------- Minibatch Grouping -------------------

def minibatch(data: List[Any], batch_size: int) -> Iterator[List[Any]]:
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]

# ------------------- Main Function -------------------

def main():
    # Check device
    device = CONFIG["device"]
    print(f"Using device: {device}")

    # Label mapping (example for NER)
    labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    # Tokenizer and model
    tokenizer = create_transformer_tokenizer(CONFIG["model_name"])
    transformer = create_transformer_model(CONFIG["model_name"])

    # Load datasets
    train_data = load_dataset(CONFIG["train_data_path"], tokenizer, label2id, CONFIG["max_length"])
    eval_data = load_dataset(CONFIG["eval_data_path"], tokenizer, label2id, CONFIG["max_length"])

    # Model
    tagger = create_transformer_tagger_model(transformer, CONFIG["num_labels"])

    # Optimizer
    optimizer = Adam(CONFIG["learning_rate"])

    # Training loop
    for epoch in range(CONFIG["epochs"]):
        np.random.shuffle(train_data)
        losses = []
        for batch in minibatch(train_data, CONFIG["batch_size"]):
            X = convert_inputs(batch)
            Y = convert_outputs(batch)
            preds, _ = tagger.begin_update(X)
            loss = ((preds - Y) ** 2).mean()
            losses.append(loss)
            # No backward for simplicity
        avg_loss = np.mean(losses)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {avg_loss:.4f}")

        # Evaluation
        all_preds, all_labels = [], []
        for batch in minibatch(eval_data, CONFIG["batch_size"]):
            X = convert_inputs(batch)
            Y = convert_outputs(batch)
            preds = tagger.predict(X)
            all_preds.append(preds)
            all_labels.append(Y)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        acc = evaluate_sequences(all_preds, all_labels)
        print(f"Validation accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
```