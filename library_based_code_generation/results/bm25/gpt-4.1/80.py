```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from datasets import load_dataset
from avalanche.benchmarks import nc_benchmark
from avalanche.training.strategies import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger

class MTDataCollator:
    def __init__(self, tokenizer, padding=True, max_length=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        batch_inputs = self.tokenizer.pad(
            {'input_ids': input_ids},
            padding=self.padding,
            max_length=self.max_length,
            return_tensors='pt'
        )
        batch_labels = self.tokenizer.pad(
            {'input_ids': labels},
            padding=self.padding,
            max_length=self.max_length,
            return_tensors='pt'
        )
        labels = batch_labels['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100
        decoder_input_ids = self._shift_right(labels)
        return {
            'input_ids': batch_inputs['input_ids'],
            'attention_mask': batch_inputs['attention_mask'],
            'labels': labels,
            'decoder_input_ids': decoder_input_ids
        }

    def _shift_right(self, labels):
        decoder_input_ids = labels.clone()
        decoder_input_ids[:, 1:] = labels[:, :-1]
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.pad_token_id
        return decoder_input_ids

class NaiveMT(Naive):
    def forward(self):
        batch = self.mbatch
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        decoder_input_ids = batch['decoder_input_ids'].to(self.device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
        return outputs

    def criterion(self):
        outputs = self.mb_output
        return outputs.loss

def preprocess_function(examples, tokenizer, max_source_length=128, max_target_length=128, src_lang="en", tgt_lang="de"):
    inputs = examples[src_lang]
    targets = examples[tgt_lang]
    model_inputs = tokenizer(
        inputs, max_length=max_source_length, truncation=True, padding='max_length'
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, truncation=True, padding='max_length'
        )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def main():
    model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("wmt14", "de-en", split={'train': 'train[:1%]', 'test': 'test[:1%]'})
    src_lang = "en"
    tgt_lang = "de"
    max_source_length = 64
    max_target_length = 64

    def preprocess(examples):
        return preprocess_function(
            examples, tokenizer, max_source_length, max_target_length, src_lang, tgt_lang
        )

    tokenized_train = dataset['train'].map(preprocess, batched=True)
    tokenized_test = dataset['test'].map(preprocess, batched=True)

    # Create Avalanche benchmark
    scenario = nc_benchmark(
        train_datasets=[tokenized_train],
        test_datasets=[tokenized_test],
        task_labels=False
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    data_collator = MTDataCollator(tokenizer, padding=True, max_length=max_source_length)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[InteractiveLogger()]
    )

    strategy = NaiveMT(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=2e-5),
        criterion=None,
        train_mb_size=8,
        train_epochs=1,
        eval_mb_size=8,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        evaluator=eval_plugin,
        data_loader=DataLoader,
        collate_fn=data_collator
    )

    for experience in scenario.train_stream:
        print(f"Start training on experience {experience.current_experience}")
        strategy.train(experience)
        print("Training completed")
        print("Computing accuracy on the test set")
        strategy.eval(scenario.test_stream)

if __name__ == "__main__":
    main()
```