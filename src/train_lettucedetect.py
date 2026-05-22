"""Fine-tune a LettuceDetect-compatible token classifier on local JSONL datasets."""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

try:
    from lettucedetect.detectors.prompt_utils import PromptUtils
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The lettucedetect package is required. Install it with `pip install lettucedetect`."
    ) from exc


DEFAULT_LETTUCE_MODEL = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
DATASET_FILES = ("clean.jsonl", "contradiction.jsonl", "overgeneration.jsonl", "missing_tool.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def resolve_dataset_paths(paths: Iterable[str]) -> list[Path]:
    resolved = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            for name in DATASET_FILES:
                candidate = path / name
                if candidate.exists():
                    resolved.append(candidate)
        elif path.exists():
            resolved.append(path)
        else:
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
    if not resolved:
        raise ValueError("No JSONL dataset files were found.")
    return resolved


def load_items(paths: list[Path]) -> list[dict]:
    items = []
    for path in paths:
        path_items = read_jsonl(path)
        for item in path_items:
            item["_source_file"] = str(path)
        items.extend(path_items)
    return items


def to_prompt(example: dict, lang: str) -> str:
    context = example.get("context") or example.get("tool_output") or ""
    question = example.get("query") or None
    return PromptUtils.format_context([context], question, lang)


def tokenize_example(example: dict, tokenizer, max_length: int, lang: str) -> dict:
    prompt = to_prompt(example, lang)
    answer = example.get("output") or example.get("model_response") or ""
    labels = example.get("hallucination_labels", [])
    gold_spans = [(int(label["start"]), int(label["end"])) for label in labels]

    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
    max_answer_tokens = max(1, max_length // 2)
    reserved_for_answer = min(len(answer_tokens), max_answer_tokens)
    prompt_max_length = max(1, max_length - reserved_for_answer - 3)
    prompt = tokenizer.decode(
        tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=prompt_max_length)[
            "input_ids"
        ],
        skip_special_tokens=True,
    )

    encoded = tokenizer(
        prompt,
        answer,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")
    sequence_ids = encoded.sequence_ids()

    token_labels = []
    for seq_id, offset in zip(sequence_ids, offsets):
        if seq_id != 1:
            token_labels.append(-100)
            continue

        start, end = offset
        if start == end:
            token_labels.append(-100)
            continue

        is_hallucinated = any(end > gold_start and start < gold_end for gold_start, gold_end in gold_spans)
        token_labels.append(1 if is_hallucinated else 0)

    encoded["labels"] = token_labels
    return encoded


def convert_dataset(items: list[dict], tokenizer, max_length: int, lang: str, desc: str) -> Dataset:
    features = [
        tokenize_example(example, tokenizer, max_length=max_length, lang=lang)
        for example in tqdm(items, desc=desc, unit="example")
    ]
    return Dataset.from_list(features)


def split_items(items: list[dict], validation_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    items = list(items)
    random.Random(seed).shuffle(items)
    valid_size = max(1, int(len(items) * validation_ratio)) if validation_ratio > 0 else 0
    valid_items = items[:valid_size]
    train_items = items[valid_size:]
    if not train_items:
        raise ValueError("Training split is empty. Use more examples or a smaller --validation_ratio.")
    return train_items, valid_items


def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    counts = {0: 0, 1: 0}
    for labels in dataset["labels"]:
        for label in labels:
            if label in counts:
                counts[label] += 1

    total = counts[0] + counts[1]
    if total == 0 or counts[0] == 0 or counts[1] == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float)

    return torch.tensor(
        [
            total / (2.0 * counts[0]),
            total / (2.0 * counts[1]),
        ],
        dtype=torch.float,
    )


class WeightedTokenTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
            ignore_index=-100,
        )
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    mask = labels != -100

    y_true = labels[mask]
    y_pred = preds[mask]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1],
        average="binary",
        zero_division=0,
    )
    accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="One or more JSONL files or directories containing clean/contradiction/overgeneration/missing_tool JSONL files.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default=DEFAULT_LETTUCE_MODEL)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--validation_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training. Recommended on CUDA.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_paths = resolve_dataset_paths(args.dataset)
    items = load_items(dataset_paths)
    train_items, valid_items = split_items(items, args.validation_ratio, args.seed)

    print(f"Loaded {len(items)} examples from {len(dataset_paths)} file(s).")
    print(f"Train examples: {len(train_items)}")
    print(f"Validation examples: {len(valid_items)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_dataset = convert_dataset(
        train_items, tokenizer, max_length=args.max_length, lang=args.lang, desc="Tokenizing train"
    )
    valid_dataset = convert_dataset(
        valid_items, tokenizer, max_length=args.max_length, lang=args.lang, desc="Tokenizing validation"
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label={0: "supported", 1: "hallucination"},
        label2id={"supported": 0, "hallucination": 1},
    )

    class_weights = None if args.no_class_weights else compute_class_weights(train_dataset)
    if class_weights is not None:
        print(f"Class weights: supported={class_weights[0]:.4f}, hallucination={class_weights[1]:.4f}")

    use_cpu = args.device == "cpu"
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("`--device cuda` was requested, but torch.cuda.is_available() is False.")
    fp16 = args.fp16 and use_cuda
    has_validation = len(valid_dataset) > 0

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="epoch" if has_validation else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=has_validation,
        metric_for_best_model="f1",
        greater_is_better=True,
        disable_tqdm=False,
        report_to="none",
        seed=args.seed,
        use_cpu=use_cpu,
        fp16=fp16,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = WeightedTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset if has_validation else None,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned LettuceDetect-compatible model to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
