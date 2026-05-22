"""LookBack Lens baseline for span-level contextual hallucination detection.

Implements the lookback-ratio features and logistic-regression classifier described in:
https://arxiv.org/abs/2407.07071

The detector uses attention from a causal LM on (query, context, answer) and predicts
hallucinated spans via a sliding-window classifier trained on local JSONL datasets.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from src.utils import merge_adjacent
except ModuleNotFoundError:
    from utils import merge_adjacent


DEFAULT_LOOKBACK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_SLIDING_WINDOW = 8
CLASSIFIER_FILENAME = "classifier.pkl"
METADATA_FILENAME = "metadata.json"


@dataclass
class LookbackLensConfig:
    model_name: str = DEFAULT_LOOKBACK_MODEL
    sliding_window: int = DEFAULT_SLIDING_WINDOW
    threshold: float = 0.5
    max_length: int = 2048


def build_prompt(query: str, context: str) -> str:
    """Format tool-calling inputs for the LLM backbone."""
    return (
        "### User Question:\n"
        f"{query.strip()}\n\n"
        "### Tool Output:\n"
        f"{context.strip()}\n\n"
        "### Answer:\n"
    )


def _resolve_device(device: str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class LookbackRatioExtractor:
    """Extract per-token lookback ratios from a causal LM forward pass."""

    def __init__(
        self,
        model_name: str = DEFAULT_LOOKBACK_MODEL,
        device: str | None = None,
        max_length: int = 2048,
        torch_dtype: torch.dtype | None = None,
    ):
        self.model_name = model_name
        self.device = _resolve_device(device)
        self.max_length = max_length
        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            attn_implementation="eager",
        )
        self.model.to(self.device)
        self.model.eval()

    def tokenize_example(self, query: str, context: str, answer: str) -> dict:
        prompt = build_prompt(query, context)
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        remaining = max(1, self.max_length - len(prompt_ids["input_ids"]))
        answer_ids = self.tokenizer(
            answer,
            add_special_tokens=False,
            truncation=True,
            max_length=remaining,
            return_offsets_mapping=True,
        )
        input_ids = prompt_ids["input_ids"] + answer_ids["input_ids"]
        attention_mask = [1] * len(input_ids)
        return {
            "prompt": prompt,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "context_length": len(prompt_ids["input_ids"]),
            "answer_offsets": answer_ids["offset_mapping"],
        }

    @torch.inference_mode()
    def extract(self, query: str, context: str, answer: str) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        encoded = self.tokenize_example(query, context, answer)
        input_ids = torch.tensor([encoded["input_ids"]], device=self.device)
        attention_mask = torch.tensor([encoded["attention_mask"]], device=self.device)
        context_length = encoded["context_length"]
        answer_len = input_ids.shape[1] - context_length
        if answer_len <= 0:
            return torch.zeros(0), encoded["answer_offsets"]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )
        num_layers = len(outputs.attentions)
        num_heads = outputs.attentions[0].shape[1]
        lookback = torch.zeros((num_layers, num_heads, answer_len), dtype=torch.float32)

        for token_idx in range(answer_len):
            pos = context_length + token_idx
            for layer_idx in range(num_layers):
                attn = outputs.attentions[layer_idx][0, :, pos, : pos + 1]
                attn_on_context = attn[:, :context_length].mean(dim=-1)
                attn_on_answer = attn[:, context_length : pos + 1].mean(dim=-1)
                lookback[layer_idx, :, token_idx] = attn_on_context / (
                    attn_on_context + attn_on_answer + 1e-8
                )

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return lookback, encoded["answer_offsets"]


def window_feature_vector(lookback: torch.Tensor, end_idx: int, sliding_window: int) -> np.ndarray:
    """Average lookback ratios inside a sliding window into a flat feature vector."""
    start_idx = max(0, end_idx - sliding_window + 1)
    window = lookback[:, :, start_idx : end_idx + 1]
    flattened = window.reshape(window.shape[0] * window.shape[1], window.shape[2]).transpose(0, 1)
    return flattened.mean(dim=0).cpu().numpy()


def token_labels_from_spans(
    answer: str,
    answer_offsets: Sequence[tuple[int, int]],
    gold_spans: Sequence[tuple[int, int]],
) -> list[int]:
    """Return per-answer-token labels: 0=hallucination, 1=factual (Lookback Lens convention)."""
    labels = []
    for start, end in answer_offsets:
        if start == end:
            continue
        overlaps = any(not (end <= gs or ge <= start) for gs, ge in gold_spans)
        labels.append(0 if overlaps else 1)
    return labels


def window_label_from_tokens(token_labels: Sequence[int], end_idx: int, sliding_window: int) -> int:
    start_idx = max(0, end_idx - sliding_window + 1)
    window = token_labels[start_idx : end_idx + 1]
    return int(min(window)) if window else 1


def spans_from_window_predictions(
    answer: str,
    answer_offsets: Sequence[tuple[int, int]],
    window_predictions: Sequence[bool],
    sliding_window: int,
) -> list[tuple[int, int]]:
    """Convert sliding-window hallucination flags into character spans over the answer."""
    valid_offsets = [(s, e) for s, e in answer_offsets if s != e]
    if not valid_offsets or not window_predictions:
        return []

    token_flags = [False] * len(valid_offsets)
    for end_idx, is_hallucinated in enumerate(window_predictions):
        if not is_hallucinated:
            continue
        start_idx = max(0, end_idx - sliding_window + 1)
        for idx in range(start_idx, end_idx + 1):
            if idx < len(token_flags):
                token_flags[idx] = True

    spans: list[tuple[int, int]] = []
    current_start = None
    current_end = None
    for (char_start, char_end), flagged in zip(valid_offsets, token_flags):
        if flagged:
            if current_start is None:
                current_start, current_end = char_start, char_end
            else:
                current_end = char_end
        elif current_start is not None:
            spans.append((current_start, current_end))
            current_start = current_end = None
    if current_start is not None:
        spans.append((current_start, current_end))
    return merge_adjacent(spans)


class LookbackLensClassifierBundle:
    """Serializable logistic-regression classifier trained on lookback-ratio features."""

    def __init__(self, classifier: LogisticRegression, config: LookbackLensConfig):
        self.classifier = classifier
        self.config = config

    def predict_window_label(self, feature_vector: np.ndarray) -> int:
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(feature_vector)[0]
            classes = list(self.classifier.classes_)
            if 0 in classes:
                hallucination_prob = proba[classes.index(0)]
                return 0 if hallucination_prob >= self.config.threshold else 1
        return int(self.classifier.predict(feature_vector)[0])

    def save(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / CLASSIFIER_FILENAME).open("wb") as f:
            pickle.dump(self.classifier, f)
        with (output_path / METADATA_FILENAME).open("w", encoding="utf8") as f:
            json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, classifier_dir: str | Path) -> "LookbackLensClassifierBundle":
        classifier_path = Path(classifier_dir)
        with (classifier_path / CLASSIFIER_FILENAME).open("rb") as f:
            classifier = pickle.load(f)
        metadata_path = classifier_path / METADATA_FILENAME
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf8") as f:
                config = LookbackLensConfig(**json.load(f))
        else:
            config = LookbackLensConfig()
        return cls(classifier=classifier, config=config)


class LookBackLensPredictor:
    """Adapter used by eval_baselines.py."""

    def __init__(
        self,
        classifier_dir: str,
        model_name: str | None = None,
        device: str | None = None,
        sliding_window: int | None = None,
        threshold: float | None = None,
    ):
        self.bundle = LookbackLensClassifierBundle.load(classifier_dir)
        config = self.bundle.config
        if model_name is not None:
            config.model_name = model_name
        if sliding_window is not None:
            config.sliding_window = sliding_window
        if threshold is not None:
            config.threshold = threshold
        self.config = config
        self.extractor = LookbackRatioExtractor(
            model_name=config.model_name,
            device=device,
            max_length=config.max_length,
        )

    def predict(self, example: dict) -> list[tuple[int, int]]:
        context = example.get("context") or example.get("tool_output") or ""
        question = example.get("query") or ""
        answer = example.get("output") or example.get("model_response") or ""
        if not answer.strip():
            return []

        lookback, answer_offsets = self.extractor.extract(question, context, answer)
        if lookback.numel() == 0:
            return []

        token_labels = token_labels_from_spans(answer, answer_offsets, [])
        window_predictions: list[bool] = []
        for end_idx in range(len(token_labels)):
            feature = window_feature_vector(lookback, end_idx, self.config.sliding_window)
            pred_label = self.bundle.predict_window_label(feature)
            window_predictions.append(pred_label == 0)

        return spans_from_window_predictions(
            answer,
            answer_offsets,
            window_predictions,
            self.config.sliding_window,
        )


def iter_training_windows(
    lookback: torch.Tensor,
    token_labels: Sequence[int],
    sliding_window: int,
) -> Iterable[tuple[np.ndarray, int]]:
    for end_idx in range(len(token_labels)):
        yield window_feature_vector(lookback, end_idx, sliding_window), window_label_from_tokens(
            token_labels, end_idx, sliding_window
        )


def build_training_matrix(examples: Iterable[dict], extractor: LookbackRatioExtractor, sliding_window: int):
    features: list[np.ndarray] = []
    labels: list[int] = []
    for example in examples:
        context = example.get("context") or example.get("tool_output") or ""
        question = example.get("query") or ""
        answer = example.get("output") or example.get("model_response") or ""
        gold_spans = [(int(item["start"]), int(item["end"])) for item in example.get("hallucination_labels", [])]
        lookback, answer_offsets = extractor.extract(question, context, answer)
        if lookback.numel() == 0:
            continue
        token_labels = token_labels_from_spans(answer, answer_offsets, gold_spans)
        for feature, label in iter_training_windows(lookback, token_labels, sliding_window):
            features.append(feature)
            labels.append(label)
    if not features:
        raise ValueError("No training windows were extracted for LookBack Lens.")
    return np.stack(features), np.array(labels, dtype=np.int64)


def train_classifier(
    examples: Iterable[dict],
    extractor: LookbackRatioExtractor,
    sliding_window: int = DEFAULT_SLIDING_WINDOW,
    threshold: float = 0.5,
) -> LookbackLensClassifierBundle:
    x_train, y_train = build_training_matrix(examples, extractor, sliding_window)
    if len(set(y_train.tolist())) < 2:
        raise ValueError(
            "LookBack Lens training needs both factual and hallucinated windows. "
            "Use more examples or include corrupted dataset files."
        )
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    classifier.fit(x_train, y_train)
    config = LookbackLensConfig(
        model_name=extractor.model_name,
        sliding_window=sliding_window,
        threshold=threshold,
        max_length=extractor.max_length,
    )
    return LookbackLensClassifierBundle(classifier=classifier, config=config)
