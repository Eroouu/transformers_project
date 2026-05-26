"""High-score span detector for tool-calling hallucination datasets.

This file is intentionally separate from the baseline scripts. It trains a
context-aware answer-token classifier, tunes the span-decoding threshold on a
held-out validation split, and can evaluate or export predictions.

Recommended first run:

python src/leaderboard_solution.py train ^
  --dataset_dir final_dataset_train ^
  --output_dir models/leaderboard_solution ^
  --base_model KRLabsOrg/lettucedect-base-modernbert-en-v1 ^
  --device cuda --fp16

Then evaluate:

python src/leaderboard_solution.py evaluate ^
  --dataset final_dataset_test ^
  --model_dir models/leaderboard_solution ^
  --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

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


DATASET_FILES = ("clean.jsonl", "contradiction.jsonl", "overgeneration.jsonl", "missing_tool.jsonl")
DEFAULT_BASE_MODEL = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
CONFIG_NAME = "leaderboard_config.json"


try:
    from lettucedetect.detectors.prompt_utils import PromptUtils
except ModuleNotFoundError:
    PromptUtils = None


@dataclass
class Example:
    item: dict
    group_id: str
    source_file: str


@dataclass
class DecodeConfig:
    threshold: float = 0.5
    max_gap_chars: int = 2
    min_span_chars: int = 1
    base_model: str = DEFAULT_BASE_MODEL
    max_length: int = 768
    lang: str = "en"


def read_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(items: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def dataset_paths(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_dir():
        found = [root / name for name in DATASET_FILES if (root / name).exists()]
        if found:
            return found
        found = sorted(root.glob("*.jsonl"))
        if found:
            return found
        raise FileNotFoundError(f"No JSONL files found in {root}")
    if root.exists():
        return [root]
    raise FileNotFoundError(f"Dataset path does not exist: {root}")


def load_examples(path: str | Path) -> list[Example]:
    paths = dataset_paths(path)

    # If this is an aligned four-file dataset, split by row id so the clean and
    # corrupted variants of the same source example stay together.
    if len(paths) == len(DATASET_FILES) and {p.name for p in paths} == set(DATASET_FILES):
        rows_by_file = {p.name: read_jsonl(p) for p in paths}
        counts = {name: len(rows) for name, rows in rows_by_file.items()}
        if len(set(counts.values())) == 1:
            examples = []
            for filename in DATASET_FILES:
                for idx, item in enumerate(rows_by_file[filename]):
                    group_id = str(item.get("example_id") or item.get("source_index") or idx)
                    examples.append(Example(item=item, group_id=group_id, source_file=filename))
            return examples

    examples = []
    for path_ in paths:
        for idx, item in enumerate(read_jsonl(path_)):
            group_id = str(item.get("example_id") or item.get("source_index") or f"{path_.name}:{idx}")
            examples.append(Example(item=item, group_id=group_id, source_file=path_.name))
    return examples


def make_prompt(example: dict, lang: str) -> str:
    context = example.get("context") or example.get("tool_output") or ""
    question = example.get("query") or ""
    if PromptUtils is not None:
        return PromptUtils.format_context([context], question or None, lang)
    return f"Question:\n{question}\n\nTool output:\n{context}\n\nAnswer:\n"


def answer_text(example: dict) -> str:
    return example.get("output") or example.get("model_response") or ""


def truncate_prompt_for_answer(prompt: str, answer: str, tokenizer, max_length: int) -> str:
    """Keep prompt short enough that answer tokens survive pair truncation."""
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    max_answer_tokens = max(1, max_length // 2)
    reserved_answer = min(len(answer_ids), max_answer_tokens)
    prompt_max_length = max(1, max_length - reserved_answer - special_tokens)
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=prompt_max_length,
    )["input_ids"]
    return tokenizer.decode(prompt_ids, skip_special_tokens=True)


def pair_truncation_mode(answer: str, tokenizer, max_length: int):
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    if len(answer_ids) <= max_length - special_tokens:
        return "only_first"
    return True


def gold_spans(example: dict) -> list[tuple[int, int]]:
    spans = []
    for label in example.get("hallucination_labels", []) or []:
        try:
            start = int(label["start"])
            end = int(label["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            spans.append((start, end))
    return spans


def spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def merge_spans(spans: Sequence[tuple[int, int]], max_gap_chars: int = 0) -> list[tuple[int, int]]:
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + max_gap_chars:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def split_by_group(examples: list[Example], validation_ratio: float, seed: int) -> tuple[list[Example], list[Example]]:
    groups = sorted({ex.group_id for ex in examples})
    random.Random(seed).shuffle(groups)
    valid_size = max(1, int(len(groups) * validation_ratio)) if validation_ratio > 0 else 0
    valid_groups = set(groups[:valid_size])
    train = [ex for ex in examples if ex.group_id not in valid_groups]
    valid = [ex for ex in examples if ex.group_id in valid_groups]
    if not train:
        raise ValueError("Training split is empty. Use a smaller --validation_ratio.")
    if not valid:
        raise ValueError("Validation split is empty. Use a larger dataset or --validation_ratio.")
    return train, valid


def tokenize_training_example(example: dict, tokenizer, max_length: int, lang: str) -> dict:
    prompt = make_prompt(example, lang)
    answer = answer_text(example)
    spans = gold_spans(example)
    prompt = truncate_prompt_for_answer(prompt, answer, tokenizer, max_length)

    encoded = tokenizer(
        prompt,
        answer,
        truncation=pair_truncation_mode(answer, tokenizer, max_length),
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")
    sequence_ids = encoded.sequence_ids()

    labels = []
    for seq_id, (start, end) in zip(sequence_ids, offsets):
        if seq_id != 1 or start == end:
            labels.append(-100)
            continue
        is_hallucinated = any(end > gold_start and start < gold_end for gold_start, gold_end in spans)
        labels.append(1 if is_hallucinated else 0)

    encoded["labels"] = labels
    return encoded


def build_dataset(examples: list[Example], tokenizer, max_length: int, lang: str, desc: str) -> Dataset:
    rows = [
        tokenize_training_example(ex.item, tokenizer, max_length=max_length, lang=lang)
        for ex in tqdm(examples, desc=desc, unit="example")
    ]
    return Dataset.from_list(rows)


def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    counts = {0: 0, 1: 0}
    for labels in dataset["labels"]:
        for label in labels:
            if label in counts:
                counts[label] += 1
    total = counts[0] + counts[1]
    if total == 0 or counts[0] == 0 or counts[1] == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    return torch.tensor(
        [total / (2.0 * counts[0]), total / (2.0 * counts[1])],
        dtype=torch.float32,
    )


class WeightedFocalTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, focal_gamma: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
            ignore_index=-100,
            reduction="none",
        )
        flat_logits = logits.view(-1, model.config.num_labels)
        flat_labels = labels.view(-1)
        token_loss = loss_fct(flat_logits, flat_labels)

        active = flat_labels != -100
        if self.focal_gamma > 0 and active.any():
            safe_labels = flat_labels.clamp(min=0)
            probs = torch.softmax(flat_logits, dim=-1)
            pt = probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1).clamp(min=1e-6, max=1.0)
            token_loss = token_loss * ((1.0 - pt) ** self.focal_gamma)

        loss = token_loss[active].mean() if active.any() else token_loss.mean()
        return (loss, outputs) if return_outputs else loss


def compute_token_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    mask = labels != -100
    if not mask.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
    y_true = labels[mask]
    y_pred = preds[mask]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1],
        average="binary",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((y_true == y_pred).mean()),
    }


class SpanPredictor:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        self.model_dir = Path(model_dir)
        self.config = load_decode_config(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def token_scores(self, example: dict) -> tuple[list[tuple[int, int]], np.ndarray]:
        prompt = make_prompt(example, self.config.lang)
        answer = answer_text(example)
        prompt = truncate_prompt_for_answer(prompt, answer, self.tokenizer, self.config.max_length)
        encoded = self.tokenizer(
            prompt,
            answer,
            truncation=pair_truncation_mode(answer, self.tokenizer, self.config.max_length),
            max_length=self.config.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        sequence_ids = encoded.sequence_ids(0)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

        answer_offsets = []
        answer_probs = []
        for seq_id, offset, prob in zip(sequence_ids, offsets, probs):
            start, end = int(offset[0]), int(offset[1])
            if seq_id == 1 and end > start:
                answer_offsets.append((start, end))
                answer_probs.append(float(prob))
        return answer_offsets, np.array(answer_probs, dtype=np.float32)

    def predict_spans(
        self,
        example: dict,
        threshold: float | None = None,
        max_gap_chars: int | None = None,
        min_span_chars: int | None = None,
    ) -> list[tuple[int, int]]:
        offsets, probs = self.token_scores(example)
        threshold = self.config.threshold if threshold is None else threshold
        max_gap_chars = self.config.max_gap_chars if max_gap_chars is None else max_gap_chars
        min_span_chars = self.config.min_span_chars if min_span_chars is None else min_span_chars
        return decode_spans(offsets, probs, threshold, max_gap_chars, min_span_chars)


def decode_spans(
    offsets: Sequence[tuple[int, int]],
    probs: np.ndarray,
    threshold: float,
    max_gap_chars: int,
    min_span_chars: int,
) -> list[tuple[int, int]]:
    raw = [offset for offset, prob in zip(offsets, probs) if prob >= threshold]
    spans = merge_spans(raw, max_gap_chars=max_gap_chars)
    return [(start, end) for start, end in spans if end - start >= min_span_chars]


def evaluate_predictions(examples: Sequence[Example], predictions: dict[int, list[tuple[int, int]]]) -> dict:
    tp = 0
    fp = 0
    fn = 0
    for idx, ex in enumerate(examples):
        gold = gold_spans(ex.item)
        pred = predictions[idx]
        used = set()
        for pred_span in pred:
            matched = False
            for gold_idx, gold_span in enumerate(gold):
                if spans_overlap(pred_span, gold_span):
                    matched = True
                    used.add(gold_idx)
                    break
            if matched:
                tp += 1
            else:
                fp += 1
        fn += len(gold) - len(used)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_examples": len(examples),
    }


def evaluate_by_type(examples: Sequence[Example], predictions: dict[int, list[tuple[int, int]]]) -> dict[str, dict]:
    by_type: dict[str, list[int]] = {}
    for idx, ex in enumerate(examples):
        ctype = ex.item.get("corruption_type") or Path(ex.source_file).stem
        by_type.setdefault(str(ctype), []).append(idx)

    result = {}
    for ctype, indices in sorted(by_type.items()):
        subset = [examples[idx] for idx in indices]
        subset_predictions = {local_idx: predictions[idx] for local_idx, idx in enumerate(indices)}
        result[ctype] = evaluate_predictions(subset, subset_predictions)
    return result


def tune_decode_config(model_dir: Path, valid_examples: list[Example], device: str | None) -> DecodeConfig:
    predictor = SpanPredictor(model_dir, device=device)
    scored = []
    for ex in tqdm(valid_examples, desc="Scoring validation", unit="example"):
        offsets, probs = predictor.token_scores(ex.item)
        scored.append((offsets, probs))

    best_config = predictor.config
    best_metrics = None
    thresholds = [round(x, 2) for x in np.linspace(0.01, 0.99, 99)]
    gap_values = [0, 1, 2, 4, 8]
    min_lengths = [1, 2, 4]

    for threshold in thresholds:
        for gap in gap_values:
            for min_len in min_lengths:
                preds = {
                    idx: decode_spans(offsets, probs, threshold, gap, min_len)
                    for idx, (offsets, probs) in enumerate(scored)
                }
                metrics = evaluate_predictions(valid_examples, preds)
                if best_metrics is None:
                    best_metrics = metrics
                    best_config.threshold = threshold
                    best_config.max_gap_chars = gap
                    best_config.min_span_chars = min_len
                    continue
                old_key = (best_metrics["f1"], best_metrics["precision"], best_metrics["recall"], -best_metrics["fp"])
                new_key = (metrics["f1"], metrics["precision"], metrics["recall"], -metrics["fp"])
                if new_key > old_key:
                    best_metrics = metrics
                    best_config.threshold = threshold
                    best_config.max_gap_chars = gap
                    best_config.min_span_chars = min_len

    print(
        "Best validation decoding: "
        f"threshold={best_config.threshold:.2f}, "
        f"max_gap_chars={best_config.max_gap_chars}, "
        f"min_span_chars={best_config.min_span_chars}, "
        f"F1={best_metrics['f1']:.4f}, P={best_metrics['precision']:.4f}, R={best_metrics['recall']:.4f}"
    )
    save_decode_config(best_config, model_dir)
    return best_config


def save_decode_config(config: DecodeConfig, model_dir: Path) -> None:
    with (model_dir / CONFIG_NAME).open("w", encoding="utf8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)


def load_decode_config(model_dir: Path) -> DecodeConfig:
    path = model_dir / CONFIG_NAME
    if not path.exists():
        return DecodeConfig()
    with path.open("r", encoding="utf8") as f:
        return DecodeConfig(**json.load(f))


def print_metrics(title: str, metrics: dict) -> None:
    print(
        f"{title}: TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
        f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
        f"N={metrics['n_examples']}"
    )


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples = load_examples(args.dataset_dir)
    train_examples, valid_examples = split_by_group(examples, args.validation_ratio, args.seed)
    print(f"Loaded {len(examples)} examples")
    print(f"Train examples: {len(train_examples)}")
    print(f"Validation examples: {len(valid_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    train_dataset = build_dataset(train_examples, tokenizer, args.max_length, args.lang, "Tokenizing train")
    valid_dataset = build_dataset(valid_examples, tokenizer, args.max_length, args.lang, "Tokenizing validation")

    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "supported", 1: "hallucination"},
        label2id={"supported": 0, "hallucination": 1},
        ignore_mismatched_sizes=True,
    )

    class_weights = None if args.no_class_weights else compute_class_weights(train_dataset)
    if class_weights is not None:
        print(f"Class weights: supported={class_weights[0]:.4f}, hallucination={class_weights[1]:.4f}")

    output_dir = Path(args.output_dir)
    use_cpu = args.device == "cpu"
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("`--device cuda` was requested, but CUDA is not available.")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
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
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        disable_tqdm=False,
        report_to="none",
        seed=args.seed,
        use_cpu=use_cpu,
        fp16=args.fp16 and use_cuda,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = WeightedFocalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_token_metrics,
        class_weights=class_weights,
        focal_gamma=args.focal_gamma,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    save_decode_config(
        DecodeConfig(
            threshold=0.5,
            max_gap_chars=2,
            min_span_chars=1,
            base_model=args.base_model,
            max_length=args.max_length,
            lang=args.lang,
        ),
        output_dir,
    )
    tune_decode_config(output_dir, valid_examples, device=args.device)
    print(f"Saved leaderboard solution to {output_dir.resolve()}")


def predict_examples(
    model_dir: str | Path,
    examples: list[Example],
    device: str | None,
) -> dict[int, list[tuple[int, int]]]:
    predictor = SpanPredictor(model_dir, device=device)
    predictions = {}
    for idx, ex in enumerate(tqdm(examples, desc="Predicting", unit="example")):
        predictions[idx] = predictor.predict_spans(ex.item)
    return predictions


def evaluate(args: argparse.Namespace) -> None:
    examples = load_examples(args.dataset)
    predictions = predict_examples(args.model_dir, examples, args.device)
    overall = evaluate_predictions(examples, predictions)
    print_metrics("Overall", overall)

    by_type = evaluate_by_type(examples, predictions)
    for ctype, metrics in by_type.items():
        print_metrics(ctype, metrics)

    if args.metrics_out:
        path = Path(args.metrics_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf8") as f:
            json.dump({"overall": overall, "by_type": by_type}, f, ensure_ascii=False, indent=2)


def export_predictions(args: argparse.Namespace) -> None:
    examples = load_examples(args.input)
    predictions = predict_examples(args.model_dir, examples, args.device)

    output_rows = []
    for idx, ex in enumerate(examples):
        item = dict(ex.item)
        labels = []
        output = answer_text(item)
        for start, end in predictions[idx]:
            labels.append(
                {
                    "start": start,
                    "end": end,
                    "label": "hallucination",
                    "type": item.get("corruption_type", "hallucination"),
                    "text": output[start:end],
                }
            )
        item["pred_hallucination_labels"] = labels
        output_rows.append(item)

    write_jsonl(output_rows, Path(args.output))
    print(f"Wrote predictions for {len(output_rows)} examples to {Path(args.output).resolve()}")


def add_train_args(subparsers) -> None:
    parser = subparsers.add_parser("train", help="Train and tune the leaderboard solution.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.08)
    parser.add_argument("--validation_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument("--lang", default="en")
    parser.set_defaults(func=train)


def add_evaluate_args(subparsers) -> None:
    parser = subparsers.add_parser("evaluate", help="Evaluate a trained solution on JSONL or dataset directory.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--metrics_out")
    parser.set_defaults(func=evaluate)


def add_predict_args(subparsers) -> None:
    parser = subparsers.add_parser("predict", help="Export predicted hallucination spans.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.set_defaults(func=export_predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate the proposed leaderboard hallucination detector.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_train_args(subparsers)
    add_evaluate_args(subparsers)
    add_predict_args(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
