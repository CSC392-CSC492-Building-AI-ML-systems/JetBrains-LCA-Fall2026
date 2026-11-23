```python
import os
import torch
import thinc
from thinc.api import Model, chain, with_getitem, list2ragged, Ragged, Optimizer, Adam, Config
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Iterable
from transformers import AutoTokenizer, AutoModel, AutoConfig
from datasets import load_dataset
from thinc.types import Floats2d, Ints2d

@dataclass
class TransformerBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor = None

def create_transformer_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)

def create_transformer_model(model_name: str):
    config = AutoConfig.from_pretrained(model_name, output_hidden_states=False)
    return AutoModel.from_pretrained(model_name, config=config)

def convert_inputs(batch: Dict[str, Any]) -> TransformerBatch:
    input_ids = torch.tensor(batch['input_ids'])
    attention_mask = torch.tensor(batch['attention_mask'])
    token_type_ids = torch.tensor(batch['token_type_ids']) if 'token_type_ids' in batch else None
    return TransformerBatch(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

def convert_outputs(outputs: torch.Tensor) -> Floats2d:
    return outputs.detach().cpu().numpy()

def group_minibatches(pairs: List[Tuple[Any, Any]], batch_size: int) -> Iterable[List[Tuple[Any, Any]]]:
    for i in range(0, len(pairs), batch_size):
        yield pairs[i:i+batch_size]

def evaluate_sequences(model: Model, data: List[Tuple[Any, Any]]) -> float:
    correct = 0
    total = 0
    for X, Y in data:
        pred = model.predict_one(X)
        correct += (pred == Y).sum()
        total += len(Y)
    return correct / total if total > 0 else 0.0

def create_transformer_tagger_model(model_name: str, n_tags: int) -> Model:
    transformer = create_transformer_model(model_name)
    def forward(model: Model, X: TransformerBatch, is_train: bool):
        outputs = transformer(
            input_ids=X.input_ids,
            attention_mask=X.attention_mask,
            token_type_ids=X.token_type_ids
        )
        last_hidden = outputs.last_hidden_state
        logits = model.get_param("W")(last_hidden)
        return logits, lambda d_logits: None
    W = thinc.api.Param(torch.nn.Linear(transformer.config.hidden_size, n_tags))
    return Model("transformer_tagger", forward, params={"W": W})

def main():
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    config = {
        "model_name": "distilbert-base-uncased",
        "n_tags": 9,
        "optimizer": "Adam",
        "learning_rate": 2e-5,
        "epochs": 3,
        "batch_size": 16,
        "dataset": "conll2003",
        "split": "train"
    }

    # Load tokenizer and dataset
    tokenizer = create_transformer_tokenizer(config["model_name"])
    dataset = load_dataset(config["dataset"], split=config["split"])

    # Prepare data
    def encode_batch(batch):
        return tokenizer.batch_encode_plus(
            batch["tokens"],
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
    encoded_dataset = dataset.map(encode_batch, batched=True)
    X = [convert_inputs({k: v[i] for k, v in encoded_dataset[i].items() if k in ["input_ids", "attention_mask", "token_type_ids"]}) for i in range(len(encoded_dataset))]
    Y = [encoded_dataset[i]["ner_tags"] for i in range(len(encoded_dataset))]
    data = list(zip(X, Y))

    # Initialize model
    model = create_transformer_tagger_model(config["model_name"], config["n_tags"])
    model.to(device)

    # Optimizer
    optimizer = Adam(model, config["learning_rate"])

    # Training loop
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        total_loss = 0.0
        for batch in group_minibatches(data, config["batch_size"]):
            X_batch, Y_batch = zip(*batch)
            logits, backprop = model.begin_update(X_batch)
            logits = torch.stack([logit for logit in logits])
            Y_tensor = torch.tensor(Y_batch, device=device)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, config["n_tags"]), Y_tensor.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Loss: {total_loss:.4f}")

        # Evaluation
        acc = evaluate_sequences(model, data[:100])
        print(f"Eval accuracy (first 100): {acc:.4f}")

if __name__ == "__main__":
    main()
```