```python
import os
import torch
import thinc
from thinc.api import Model, chain, with_getitem, list2ragged, Ragged, Adam, Config
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Iterable
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import logging as hf_logging
from datasets import load_dataset
from tqdm import tqdm

hf_logging.set_verbosity_error()

@dataclass
class TransformerBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor = None

def create_transformer_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)

def create_transformer_model(model_name: str):
    return AutoModel.from_pretrained(model_name)

def batch_encode(tokenizer, texts: List[List[str]], max_length: int = 128) -> TransformerBatch:
    encodings = tokenizer(
        texts,
        is_split_into_words=True,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    )
    return TransformerBatch(
        input_ids=encodings['input_ids'],
        attention_mask=encodings['attention_mask'],
        token_type_ids=encodings.get('token_type_ids')
    )

def transformer_inputs(batch: TransformerBatch) -> Dict[str, torch.Tensor]:
    inputs = {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask
    }
    if batch.token_type_ids is not None:
        inputs["token_type_ids"] = batch.token_type_ids
    return inputs

def transformer_outputs(outputs) -> torch.Tensor:
    # outputs[0] is last_hidden_state
    return outputs[0]

def create_transformer_tagger_model(model_name: str, n_tags: int) -> Model:
    transformer = create_transformer_model(model_name)
    def forward(model, X, is_train):
        inputs = transformer_inputs(X)
        outputs = transformer(**inputs)
        hidden = transformer_outputs(outputs)
        logits = model.get_ref("tag_proj")(hidden)
        return logits, lambda d_logits: d_logits
    tag_proj = thinc.api.Linear(n_tags)
    return Model(
        "transformer_tagger",
        forward,
        refs={"tag_proj": tag_proj}
    )

def evaluate_sequences(model: Model, dataset: List[Tuple[List[str], List[int]]], tokenizer, label_map: Dict[int, str]) -> float:
    correct = 0
    total = 0
    for tokens, labels in dataset:
        batch = batch_encode(tokenizer, [tokens])
        logits = model.predict(batch)
        preds = logits.argmax(-1).squeeze(0).tolist()
        for pred, gold in zip(preds, labels):
            if pred == gold:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0

def minibatch_pairs(data: List[Tuple[List[str], List[int]]], batch_size: int) -> Iterable[List[Tuple[List[str], List[int]]]]:
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def load_data(tokenizer, label2id: Dict[str, int], split: str = "train", max_samples: int = 1000):
    dataset = load_dataset("conll2003", split=split)
    data = []
    for example in dataset.select(range(min(max_samples, len(dataset)))):
        tokens = example["tokens"]
        labels = [label2id.get(l, 0) for l in example["ner_tags"]]
        data.append((tokens, labels))
    return data

def main():
    # Configuration
    config = {
        "model_name": "distilbert-base-uncased",
        "n_tags": 9,  # For CoNLL2003 NER
        "optimizer": "Adam",
        "learning_rate": 2e-5,
        "epochs": 3,
        "batch_size": 16,
        "max_length": 128,
        "max_train_samples": 512,
        "max_eval_samples": 128
    }

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and label map
    tokenizer = create_transformer_tokenizer(config["model_name"])
    label_list = [
        "O", "B-PER", "I-PER", "B-ORG", "I-ORG",
        "B-LOC", "I-LOC", "B-MISC", "I-MISC"
    ]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    # Load data
    train_data = load_data(tokenizer, label2id, split="train", max_samples=config["max_train_samples"])
    eval_data = load_data(tokenizer, label2id, split="validation", max_samples=config["max_eval_samples"])

    # Initialize model
    model = create_transformer_tagger_model(config["model_name"], config["n_tags"])
    model.to(device)

    # Optimizer
    optimizer = Adam(model, learn_rate=config["learning_rate"])

    # Training loop
    for epoch in range(config["epochs"]):
        model.train = True
        total_loss = 0.0
        for batch in tqdm(minibatch_pairs(train_data, config["batch_size"]), desc=f"Epoch {epoch+1}"):
            tokens_batch, labels_batch = zip(*batch)
            batch_enc = batch_encode(tokenizer, list(tokens_batch), max_length=config["max_length"])
            batch_enc = TransformerBatch(
                input_ids=batch_enc.input_ids.to(device),
                attention_mask=batch_enc.attention_mask.to(device),
                token_type_ids=batch_enc.token_type_ids.to(device) if batch_enc.token_type_ids is not None else None
            )
            logits, backprop = model.begin_update(batch_enc, is_train=True)
            logits = logits.view(-1, config["n_tags"])
            labels = torch.tensor([l for labels in labels_batch for l in labels], dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            backprop(torch.autograd.grad(loss, logits)[0])
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

        # Evaluation
        model.train = False
        acc = evaluate_sequences(model, eval_data, tokenizer, id2label)
        print(f"Epoch {epoch+1} Eval Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
```