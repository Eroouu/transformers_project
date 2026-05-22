"""Starter training script for a span-level token-classification model.

This script provides a small pipeline to convert the RAGTruth-style JSONL into token
classification labels (B/I/O) and fine-tune a Hugging Face Transformer. It's a minimal
example and meant as a starting point for experiments.
"""
import argparse
import json
from typing import List

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)


def load_jsonl(path):
    items = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            items.append(json.loads(line))
    return items


def char_spans_to_token_labels(example, tokenizer, max_length=512):
    text = example['output']
    anns = example.get('hallucination_labels', [])
    tokenized = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_length)
    offsets = tokenized.pop('offset_mapping')
    labels = [0] * len(offsets)  # 0=O, 1=B-H, 2=I-H
    gold_spans = [(a['start'], a['end']) for a in anns]
    for i, (s, e) in enumerate(offsets):
        for gs, ge in gold_spans:
            if not (e <= gs or ge <= s):
                # overlap
                if labels[i] == 0:
                    labels[i] = 1
                else:
                    labels[i] = 2
    tokenized['labels'] = labels
    return tokenized


def convert_dataset(items: List[dict], tokenizer_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    features = [
        char_spans_to_token_labels(ex, tokenizer)
        for ex in tqdm(items, desc='Tokenizing examples', unit='example')
    ]
    return Dataset.from_list(features), tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model', default='bert-base-uncased')
    args = parser.parse_args()

    items = load_jsonl(args.dataset)
    ds, tokenizer = convert_dataset(items, tokenizer_name=args.model)

    model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=3)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=4,
        logging_steps=10,
        logging_strategy='steps',
        save_strategy='no',
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    main()
