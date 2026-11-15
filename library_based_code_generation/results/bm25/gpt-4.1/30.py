```python
import os
import torch
import thinc
from thinc.api import Model, chain, with_getitem, list2ragged, Ragged, Optimizer, Adam, set_dropout_rate
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Iterator
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

@dataclass
class BatchEncodingOutput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    labels: torch.Tensor

def create_transformer_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)

def create_transformer_model(model_name: str):
    return AutoModel.from_pretrained(model_name)

def batch_encode(tokenizer, texts: List[List[str]], labels: List[List[int]], max_length: int = 128):
    encodings = tokenizer(
        texts,
        is_split_into_words=True,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    )
    # Align labels with word_ids
    encoded_labels = []
    for i, label in enumerate(labels):
        word_ids = encodings.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        encoded_labels.append(label_ids)
    encodings['labels'] = torch.tensor(encoded_labels)
    return BatchEncodingOutput(
        input_ids=encodings['input_ids'],
        attention_mask=encodings['attention_mask'],
        token_type_ids=encodings.get('token_type_ids', torch.zeros_like(encodings['input_ids'])),
        labels=encodings['labels']
    )

def transformer_inputs(batch: BatchEncodingOutput):
    return {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "token_type_ids": batch.token_type_ids
    }

def transformer_outputs(outputs, batch: BatchEncodingOutput):
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1)
    return preds, batch.labels

def evaluate_sequences(preds: torch.Tensor, labels: torch.Tensor) -> float:
    mask = labels != -100
    correct = (preds[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0

def group_minibatches(data: List[Tuple[List[str], List[int]]], batch_size: int) -> Iterator[List[Tuple[List[str], List[int]]]]:
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def create_transformer_tagger_model(model_name: str, num_labels: int, dropout: float = 0.1) -> Model:
    class TransformerTagger(thinc.Model):
        def __init__(self, name, transformer, num_labels, dropout):
            super().__init__(name)
            self.transformer = transformer
            self.dropout = torch.nn.Dropout(dropout)
            self.classifier = torch.nn.Linear(transformer.config.hidden_size, num_labels)

        def forward(self, X, is_train):
            outputs = self.transformer(
                input_ids=X['input_ids'],
                attention_mask=X['attention_mask'],
                token_type_ids=X['token_type_ids']
            )
            sequence_output = outputs.last_hidden_state
            sequence_output = self.dropout(sequence_output)
            logits = self.classifier(sequence_output)
            return logits, lambda dY: None

    transformer = create_transformer_model(model_name)
    return Model(
        "transformer_tagger",
        TransformerTagger("transformer_tagger", transformer, num_labels, dropout)
    )

def load_dataset():
    # Dummy dataset: List of (tokens, labels)
    # Replace with actual dataset loading logic
    data = [
        (["Hello", "world", "!"], [0, 1, 2]),
        (["I", "love", "Python", "."], [0, 1, 2, 2]),
        (["Transformers", "are", "awesome", "."], [0, 1, 2, 2]),
    ]
    return data

def main():
    # Configuration
    config = {
        "model_name": "distilbert-base-uncased",
        "num_labels": 3,
        "dropout": 0.1,
        "learning_rate": 5e-5,
        "epochs": 3,
        "batch_size": 2,
        "max_length": 32,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    print(f"Using device: {config['device']}")

    tokenizer = create_transformer_tokenizer(config["model_name"])
    model = create_transformer_tagger_model(
        config["model_name"], config["num_labels"], config["dropout"]
    )
    model.to(config["device"])

    optimizer = Adam(model, config["learning_rate"])

    dataset = load_dataset()
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        total_loss = 0.0
        total_acc = 0.0
        total_batches = 0
        for batch_data in group_minibatches(dataset, config["batch_size"]):
            texts, labels = zip(*batch_data)
            batch = batch_encode(tokenizer, list(texts), list(labels), max_length=config["max_length"])
            batch = BatchEncodingOutput(
                input_ids=batch.input_ids.to(config["device"]),
                attention_mask=batch.attention_mask.to(config["device"]),
                token_type_ids=batch.token_type_ids.to(config["device"]),
                labels=batch.labels.to(config["device"])
            )
            model.train()
            optimizer.zero_grad()
            logits, _ = model(transformer_inputs(batch), is_train=True)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, config["num_labels"]),
                batch.labels.view(-1)
            )
            loss.backward()
            optimizer.step()
            preds = torch.argmax(logits, dim=-1)
            acc = evaluate_sequences(preds, batch.labels)
            total_loss += loss.item()
            total_acc += acc
            total_batches += 1
        print(f"Loss: {total_loss/total_batches:.4f} | Accuracy: {total_acc/total_batches:.4f}")

if __name__ == "__main__":
    main()
```