```python
import os
import torch
import thinc
from thinc.api import Model, chain, with_getitem, list2ragged, Ragged, Adam, Config, require_gpu, set_gpu_allocator
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Iterator
from transformers import AutoTokenizer, AutoModel, AutoConfig
from datasets import load_dataset

@dataclass
class TransformerBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    labels: torch.Tensor

def create_transformer_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)

def create_transformer_model(model_name: str):
    return AutoModel.from_pretrained(model_name)

def create_transformer_tagger_model(model_name: str, n_tags: int) -> Model:
    import torch.nn as nn
    class TransformerTagger(nn.Module):
        def __init__(self, model_name, n_tags):
            super().__init__()
            self.transformer = AutoModel.from_pretrained(model_name)
            hidden_size = self.transformer.config.hidden_size
            self.classifier = nn.Linear(hidden_size, n_tags)

        def forward(self, input_ids, attention_mask, token_type_ids):
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            sequence_output = outputs.last_hidden_state
            logits = self.classifier(sequence_output)
            return logits

    tagger = TransformerTagger(model_name, n_tags)
    return thinc.api.PyTorchWrapper(tagger)

def convert_transformer_inputs(batch: Dict[str, Any], device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(batch["input_ids"]).to(device)
    attention_mask = torch.tensor(batch["attention_mask"]).to(device)
    token_type_ids = torch.tensor(batch.get("token_type_ids", [[0]*len(ids) for ids in batch["input_ids"]])).to(device)
    labels = torch.tensor(batch["labels"]).to(device)
    return input_ids, attention_mask, token_type_ids, labels

def convert_transformer_outputs(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)

def evaluate_sequences(preds: torch.Tensor, labels: torch.Tensor) -> float:
    correct = (preds == labels).float()
    mask = (labels != -100).float()
    accuracy = (correct * mask).sum() / mask.sum()
    return accuracy.item()

def group_minibatches(dataset, batch_size: int) -> Iterator[Dict[str, Any]]:
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        yield {
            "input_ids": [item["input_ids"] for item in batch],
            "attention_mask": [item["attention_mask"] for item in batch],
            "token_type_ids": [item.get("token_type_ids", [0]*len(item["input_ids"])) for item in batch],
            "labels": [item["labels"] for item in batch]
        }

def main():
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        require_gpu(0)
        set_gpu_allocator("pytorch")

    # Configuration
    config = Config().from_str("""
    [model]
    model_name = "distilbert-base-uncased"
    n_tags = 9

    [training]
    batch_size = 8
    epochs = 3
    learning_rate = 5e-5
    """)
    model_name = config["model"]["model_name"]
    n_tags = config["model"]["n_tags"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]

    # Tokenizer and Model
    tokenizer = create_transformer_tokenizer(model_name)
    model = create_transformer_tagger_model(model_name, n_tags)
    model.to(device)

    # Optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # Load dataset (example: conll2003)
    dataset = load_dataset("conll2003", split="train[:100]")  # small subset for demo

    # Preprocess dataset
    label2id = {label: i for i, label in enumerate(dataset.features["ner_tags"].feature.names)}
    def encode_example(example):
        tokens = example["tokens"]
        labels = example["ner_tags"]
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=128
        )
        word_ids = encoding.word_ids()
        label_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(-100)
            prev_word_idx = word_idx
        encoding["labels"] = label_ids
        return encoding

    encoded_dataset = [encode_example(ex) for ex in dataset]

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        for batch in group_minibatches(encoded_dataset, batch_size):
            input_ids, attention_mask, token_type_ids, labels = convert_transformer_inputs(batch, device)
            optimizer.zero_grad()
            logits = model((input_ids, attention_mask, token_type_ids))
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, n_tags), labels.view(-1))
            loss.backward()
            optimizer.step_schedules(model)
            total_loss += loss.item()
            preds = convert_transformer_outputs(logits)
            acc = evaluate_sequences(preds, labels)
            total_acc += acc
            n_batches += 1
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/n_batches:.4f} - Acc: {total_acc/n_batches:.4f}")

if __name__ == "__main__":
    main()
```