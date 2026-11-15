```python
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from avalanche.benchmarks import nc_benchmark
from avalanche.training.strategies import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.benchmarks.utils import AvalancheDataset
from typing import Dict, Any, List


class MTDataCollator:
    def __init__(self, tokenizer, model=None, padding=True, max_length=None):
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "labels": labels},
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt"
        )
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            batch["decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
        else:
            batch["decoder_input_ids"] = batch["labels"].clone()
            batch["decoder_input_ids"][batch["decoder_input_ids"] == self.tokenizer.pad_token_id] = -100
        return batch


class NaiveMT(Naive):
    def forward(self):
        batch = self.mbatch
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        decoder_input_ids = batch.get("decoder_input_ids", None)
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.to(self.device)
        labels = batch["labels"].to(self.device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        return outputs

    def criterion(self):
        outputs = self.mb_output
        return outputs.loss

    def _unpack_minibatch(self):
        # Huggingface batch is already a dict of tensors
        self.mbatch = self.mbatch


def preprocess_function(examples, tokenizer, max_source_length=128, max_target_length=128, src_lang="en", tgt_lang="de"):
    inputs = examples["translation"][src_lang]
    targets = examples["translation"][tgt_lang]
    model_inputs = tokenizer(
        inputs, max_length=max_source_length, truncation=True, padding=False
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, truncation=True, padding=False
        )["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs


def main():
    model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    dataset = load_dataset("wmt14", "de-en", split="train[:2000]")
    val_dataset = load_dataset("wmt14", "de-en", split="validation[:500]")

    src_lang = "en"
    tgt_lang = "de"

    def preprocess(examples):
        return preprocess_function(
            examples, tokenizer, src_lang=src_lang, tgt_lang=tgt_lang
        )

    tokenized_train = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    tokenized_val = val_dataset.map(preprocess, batched=True, remove_columns=val_dataset.column_names)

    # Split into 2 experiences for demonstration
    train_chunks = tokenized_train.train_test_split(test_size=0.5, shuffle=True, seed=42)
    val_chunks = tokenized_val.train_test_split(test_size=0.5, shuffle=True, seed=42)

    train_datasets = [train_chunks["train"], train_chunks["test"]]
    val_datasets = [val_chunks["train"], val_chunks["test"]]

    # Wrap Huggingface datasets as AvalancheDataset
    avalanche_train_datasets = [
        AvalancheDataset(ds, transform=None, target_transform=None, task_labels=0)
        for ds in train_datasets
    ]
    avalanche_val_datasets = [
        AvalancheDataset(ds, transform=None, target_transform=None, task_labels=0)
        for ds in val_datasets
    ]

    scenario = nc_benchmark(
        train_datasets=avalanche_train_datasets,
        test_datasets=avalanche_val_datasets,
        n_experiences=2,
        task_labels=False,
    )

    data_collator = MTDataCollator(tokenizer, model=model, padding=True)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[]
    )

    strategy = NaiveMT(
        model,
        torch.optim.AdamW(model.parameters(), lr=2e-5),
        torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id),
        train_mb_size=8,
        train_epochs=1,
        eval_mb_size=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        evaluator=eval_plugin,
        data_loader=DataLoader,
        collate_fn=data_collator,
    )

    for experience in scenario.train_stream:
        print(f"Start training on experience {experience.current_experience}")
        strategy.train(experience.dataset)
        print("Training completed")

        print("Computing accuracy on the test set")
        strategy.eval(scenario.test_stream)

if __name__ == "__main__":
    main()
```