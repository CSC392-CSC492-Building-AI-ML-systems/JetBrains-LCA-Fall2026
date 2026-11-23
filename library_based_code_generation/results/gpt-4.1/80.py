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

class CustomDataCollator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        input_ids = self.tokenizer.pad({'input_ids': input_ids}, return_tensors='pt')['input_ids']
        labels = self.tokenizer.pad({'input_ids': labels}, return_tensors='pt')['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'decoder_input_ids': decoder_input_ids
        }

class HuggingfaceNaive(Naive):
    def forward(self):
        batch = self.mbatch
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        decoder_input_ids = batch['decoder_input_ids'].to(self.device)
        outputs = self.model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
        return outputs

    def criterion(self):
        return self.mb_output.loss

def preprocess_function(examples, tokenizer, src_lang, tgt_lang, max_length=128):
    inputs = examples[src_lang]
    targets = examples[tgt_lang]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding='max_length')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_length, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def main():
    model_name = "Helsinki-NLP/opus-mt-en-de"
    src_lang = "en"
    tgt_lang = "de"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    dataset = load_dataset("wmt14", "de-en", split={'train': 'train[:1%]', 'test': 'test[:1%]'})
    train_dataset = dataset['train'].map(
        lambda x: preprocess_function(x, tokenizer, src_lang, tgt_lang),
        batched=True
    )
    test_dataset = dataset['test'].map(
        lambda x: preprocess_function(x, tokenizer, src_lang, tgt_lang),
        batched=True
    )

    # Split train into 3 experiences for continual learning
    n_experiences = 3
    train_size = len(train_dataset)
    exp_size = train_size // n_experiences
    experiences = []
    for i in range(n_experiences):
        start = i * exp_size
        end = (i + 1) * exp_size if i < n_experiences - 1 else train_size
        experiences.append(torch.utils.data.Subset(train_dataset, range(start, end)))

    scenario = nc_benchmark(
        train_datasets=experiences,
        test_datasets=[test_dataset],
        task_labels=False
    )

    data_collator = CustomDataCollator(tokenizer, model)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[InteractiveLogger()]
    )

    strategy = HuggingfaceNaive(
        model,
        torch.optim.Adam(model.parameters(), lr=2e-5),
        torch.nn.CrossEntropyLoss(ignore_index=-100),
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
        print("End training, evaluating...")
        strategy.eval(scenario.test_stream)

if __name__ == "__main__":
    main()
```