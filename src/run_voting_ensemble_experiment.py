"""Run a weighted-voting ensemble experiment for tool-calling hallucination detection.

This script keeps existing project files unchanged and writes all artifacts into a
new run directory under outputs/ by default.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from uuid import uuid4

import numpy as np
from tqdm.auto import tqdm

try:
    from src.eval_baselines import (
        DEFAULT_LETTUCE_MODEL,
        build_predictor,
        evaluate as eval_baseline_script,
    )
    from src.lookback_lens import (
        DEFAULT_LOOKBACK_MODEL,
        DEFAULT_SLIDING_WINDOW,
        LookbackRatioExtractor,
        train_classifier,
    )
    from src.utils import merge_adjacent, read_jsonl, spans_overlap
except ModuleNotFoundError:
    from eval_baselines import (
        DEFAULT_LETTUCE_MODEL,
        build_predictor,
        evaluate as eval_baseline_script,
    )
    from lookback_lens import (
        DEFAULT_LOOKBACK_MODEL,
        DEFAULT_SLIDING_WINDOW,
        LookbackRatioExtractor,
        train_classifier,
    )
    from utils import merge_adjacent, read_jsonl, spans_overlap


DATASET_FILES = ("clean.jsonl", "contradiction.jsonl", "overgeneration.jsonl", "missing_tool.jsonl")


@dataclass
class ExampleRecord:
    uid: str
    aligned_index: int
    corruption_type: str
    item: dict


def parse_methods(value: str) -> List[str]:
    methods = [m.strip() for m in value.split(",") if m.strip()]
    if not methods:
        raise ValueError("At least one method must be provided.")
    valid = {"tool_overlap", "lettucedetect", "lookback_lens"}
    invalid = [m for m in methods if m not in valid]
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}. Valid methods: {sorted(valid)}")
    return methods


def parse_number_list(value: str, as_int: bool = False) -> List[float]:
    values: List[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part) if as_int else float(part))
    if not values:
        raise ValueError("Expected a non-empty comma-separated list.")
    return values


def resolve_output_dir(base_output: str | None, seed: int) -> Path:
    if base_output:
        return Path(base_output).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "ensemble_voting" / f"run_{timestamp}_seed{seed}"


def load_aligned_dataset(dataset_dir: str, limit_per_file: int | None = None) -> Tuple[List[ExampleRecord], int]:
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    per_file_items: Dict[str, List[dict]] = {}
    for filename in DATASET_FILES:
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        items = read_jsonl(str(path))
        if limit_per_file is not None:
            items = items[:limit_per_file]
        per_file_items[filename] = items

    counts = {name: len(items) for name, items in per_file_items.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Files are not aligned by row count: {counts}")

    n_rows = next(iter(counts.values()))
    all_examples: List[ExampleRecord] = []
    for filename, items in per_file_items.items():
        corruption_type = filename.replace(".jsonl", "")
        for idx, item in enumerate(items):
            uid = f"{corruption_type}:{idx}"
            all_examples.append(
                ExampleRecord(
                    uid=uid,
                    aligned_index=idx,
                    corruption_type=corruption_type,
                    item=item,
                )
            )

    return all_examples, n_rows


def build_splits(
    n_rows: int,
    seed: int,
    test_ratio: float,
    tune_ratio: float,
) -> Dict[str, set]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("--test_ratio must be in (0, 1).")
    if not (0.0 < tune_ratio < 1.0):
        raise ValueError("--tune_ratio must be in (0, 1).")

    indices = list(range(n_rows))
    random.Random(seed).shuffle(indices)

    test_size = max(1, int(n_rows * test_ratio))
    test_indices = set(indices[:test_size])
    non_test = indices[test_size:]
    if len(non_test) < 2:
        raise ValueError("Not enough non-test rows to reserve a tuning split.")

    tune_size = max(1, int(len(non_test) * tune_ratio))
    if tune_size >= len(non_test):
        tune_size = len(non_test) - 1

    tune_indices = set(non_test[:tune_size])
    train_indices = set(non_test[tune_size:])
    return {
        "train": train_indices,
        "tune": tune_indices,
        "test": test_indices,
        "non_test": train_indices.union(tune_indices),
    }


def split_examples(examples: Sequence[ExampleRecord], split_indices: Dict[str, set]) -> Dict[str, List[ExampleRecord]]:
    split_examples_map: Dict[str, List[ExampleRecord]] = {"train": [], "tune": [], "test": [], "non_test": []}
    for ex in examples:
        idx = ex.aligned_index
        if idx in split_indices["train"]:
            split_examples_map["train"].append(ex)
            split_examples_map["non_test"].append(ex)
        elif idx in split_indices["tune"]:
            split_examples_map["tune"].append(ex)
            split_examples_map["non_test"].append(ex)
        elif idx in split_indices["test"]:
            split_examples_map["test"].append(ex)
        else:
            raise RuntimeError(f"Aligned index {idx} was not assigned to any split.")
    return split_examples_map


def sanitize_spans(spans: Iterable[Sequence[int]], text_len: int) -> List[Tuple[int, int]]:
    cleaned: List[Tuple[int, int]] = []
    for span in spans or []:
        if len(span) < 2:
            continue
        start = max(0, min(text_len, int(span[0])))
        end = max(0, min(text_len, int(span[1])))
        if end > start:
            cleaned.append((start, end))
    return merge_adjacent(cleaned)


def spans_to_mask(spans: Sequence[Tuple[int, int]], text_len: int) -> np.ndarray:
    mask = np.zeros(text_len, dtype=np.uint8)
    for start, end in spans:
        mask[start:end] = 1
    return mask


def mask_to_spans(mask: np.ndarray) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    if mask.size == 0:
        return spans
    in_span = False
    start = 0
    for idx, value in enumerate(mask.tolist()):
        if value and not in_span:
            in_span = True
            start = idx
        elif not value and in_span:
            spans.append((start, idx))
            in_span = False
    if in_span:
        spans.append((start, int(mask.size)))
    return spans


def evaluate_from_predictions(
    examples: Sequence[ExampleRecord],
    predictions: Dict[str, List[Tuple[int, int]]],
) -> Dict[str, float]:
    tp = 0
    fp = 0
    fn = 0
    for ex in examples:
        gold = ex.item.get("hallucination_labels", [])
        gold_spans = [(int(g["start"]), int(g["end"])) for g in gold]
        pred_spans = predictions[ex.uid]

        used = set()
        for pred in pred_spans:
            matched = False
            for idx, gold_span in enumerate(gold_spans):
                if spans_overlap(pred, gold_span):
                    matched = True
                    used.add(idx)
                    break
            if matched:
                tp += 1
            else:
                fp += 1
        fn += len(gold_spans) - len(used)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_examples": len(examples),
    }


def evaluate_per_type(
    examples: Sequence[ExampleRecord],
    predictions: Dict[str, List[Tuple[int, int]]],
) -> Dict[str, Dict[str, float]]:
    by_type: Dict[str, List[ExampleRecord]] = {}
    for ex in examples:
        by_type.setdefault(ex.corruption_type, []).append(ex)
    return {ctype: evaluate_from_predictions(subset, predictions) for ctype, subset in by_type.items()}


def train_lookback_on_non_test(
    non_test_examples: Sequence[ExampleRecord],
    classifier_dir: Path,
    model_name: str,
    device: str | None,
    sliding_window: int,
    threshold: float,
    max_length: int,
):
    classifier_dir.mkdir(parents=True, exist_ok=True)
    extractor = LookbackRatioExtractor(
        model_name=model_name,
        device=device,
        max_length=max_length,
    )
    bundle = train_classifier(
        tqdm((ex.item for ex in non_test_examples), total=len(non_test_examples), desc="Training LookBack Lens"),
        extractor=extractor,
        sliding_window=sliding_window,
        threshold=threshold,
    )
    bundle.save(classifier_dir)


def generate_predictions(
    method: str,
    examples_by_split: Dict[str, List[ExampleRecord]],
    lettuce_model: str,
    device: str | None,
    lookback_classifier_dir: str,
    lookback_model: str | None,
    lookback_sliding_window: int | None,
    lookback_threshold: float | None,
) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
    predictor = build_predictor(
        method,
        lettuce_model=lettuce_model,
        device=device,
        lookback_classifier_dir=lookback_classifier_dir,
        lookback_model=lookback_model,
        lookback_sliding_window=lookback_sliding_window,
        lookback_threshold=lookback_threshold,
    )

    results: Dict[str, Dict[str, List[Tuple[int, int]]]] = {"train": {}, "tune": {}, "test": {}, "non_test": {}}
    for split_name, split_examples_ in examples_by_split.items():
        for ex in tqdm(split_examples_, desc=f"Predicting {method} on {split_name}", unit="example"):
            output_text = ex.item.get("output") or ex.item.get("model_response") or ""
            raw = predictor(ex.item)
            spans = sanitize_spans(raw, len(output_text))
            results[split_name][ex.uid] = spans
    return results


def build_masks_for_method_predictions(
    examples: Sequence[ExampleRecord],
    predictions: Dict[str, List[Tuple[int, int]]],
) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    for ex in examples:
        output_text = ex.item.get("output") or ex.item.get("model_response") or ""
        spans = predictions[ex.uid]
        masks[ex.uid] = spans_to_mask(spans, len(output_text))
    return masks


def ensemble_predict_for_examples(
    examples: Sequence[ExampleRecord],
    method_masks: Dict[str, Dict[str, np.ndarray]],
    method_weights: Dict[str, int],
    threshold: float,
) -> Dict[str, List[Tuple[int, int]]]:
    predictions: Dict[str, List[Tuple[int, int]]] = {}
    for ex in examples:
        output_text = ex.item.get("output") or ex.item.get("model_response") or ""
        if not output_text:
            predictions[ex.uid] = []
            continue
        score = np.zeros(len(output_text), dtype=np.int32)
        for method, weight in method_weights.items():
            if weight <= 0:
                continue
            score += weight * method_masks[method][ex.uid]
        pred_mask = (score >= threshold).astype(np.uint8)
        predictions[ex.uid] = mask_to_spans(pred_mask)
    return predictions


def run_ensemble_grid_search(
    tune_examples: Sequence[ExampleRecord],
    method_masks: Dict[str, Dict[str, np.ndarray]],
    methods: Sequence[str],
    weight_values: Sequence[int],
    threshold_values: Sequence[float] | None,
) -> Dict[str, object]:
    best: Dict[str, object] | None = None
    for weights_tuple in itertools.product(weight_values, repeat=len(methods)):
        if sum(weights_tuple) <= 0:
            continue
        method_weights = dict(zip(methods, weights_tuple))
        if threshold_values is None:
            thresholds = [float(v) for v in range(1, int(sum(weights_tuple)) + 1)]
        else:
            thresholds = list(threshold_values)

        for threshold in thresholds:
            preds = ensemble_predict_for_examples(
                examples=tune_examples,
                method_masks=method_masks,
                method_weights=method_weights,
                threshold=threshold,
            )
            metrics = evaluate_from_predictions(tune_examples, preds)
            candidate = {
                "method_weights": method_weights,
                "threshold": threshold,
                "metrics": metrics,
            }
            if best is None:
                best = candidate
                continue
            old = best["metrics"]
            new = metrics
            if (
                (new["f1"], new["precision"], new["recall"], -new["fp"])
                > (old["f1"], old["precision"], old["recall"], -old["fp"])
            ):
                best = candidate

    if best is None:
        raise RuntimeError("Ensemble grid search failed to produce any candidate.")
    return best


def check_prediction_counts(
    methods: Sequence[str],
    examples_by_split: Dict[str, List[ExampleRecord]],
    prediction_store: Dict[str, Dict[str, Dict[str, List[Tuple[int, int]]]]],
):
    for method in methods:
        for split_name, split_examples_ in examples_by_split.items():
            n_examples = len(split_examples_)
            n_preds = len(prediction_store[method][split_name])
            if n_examples != n_preds:
                raise AssertionError(
                    f"Prediction count mismatch: method={method}, split={split_name}, "
                    f"examples={n_examples}, preds={n_preds}"
                )


def check_mask_roundtrip(masks: Dict[str, Dict[str, np.ndarray]]):
    for method, per_example in masks.items():
        for uid, mask in per_example.items():
            roundtrip = spans_to_mask(mask_to_spans(mask), len(mask))
            if not np.array_equal(mask, roundtrip):
                raise AssertionError(f"Mask roundtrip mismatch for method={method}, example={uid}")


def check_metric_parity_with_eval_baselines(
    tune_examples: Sequence[ExampleRecord],
    local_predictions: Dict[str, List[Tuple[int, int]]],
    workspace_tmp_dir: Path,
):
    workspace_tmp_dir.mkdir(parents=True, exist_ok=True)
    path = workspace_tmp_dir / f"subset_{uuid4().hex}.jsonl"
    try:
        with path.open("w", encoding="utf8") as f:
            for ex in tune_examples:
                f.write(json.dumps(ex.item, ensure_ascii=False) + "\n")

        script_metrics = eval_baseline_script(str(path), method="tool_overlap")
        local_metrics = evaluate_from_predictions(tune_examples, local_predictions)

        keys = ("tp", "fp", "fn")
        for key in keys:
            if int(script_metrics[key]) != int(local_metrics[key]):
                raise AssertionError(
                    f"Metric parity failed on {key}: eval_baselines={script_metrics[key]} local={local_metrics[key]}"
                )
        float_keys = ("precision", "recall", "f1")
        for key in float_keys:
            if abs(float(script_metrics[key]) - float(local_metrics[key])) > 1e-12:
                raise AssertionError(
                    f"Metric parity failed on {key}: eval_baselines={script_metrics[key]} local={local_metrics[key]}"
                )
    finally:
        if path.exists():
            path.unlink()
        try:
            if workspace_tmp_dir.exists() and not any(workspace_tmp_dir.iterdir()):
                workspace_tmp_dir.rmdir()
        except OSError:
            pass


def save_json(data: object, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_prediction_artifacts(
    output_dir: Path,
    methods: Sequence[str],
    examples_by_split: Dict[str, List[ExampleRecord]],
    baseline_predictions: Dict[str, Dict[str, Dict[str, List[Tuple[int, int]]]]],
):
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for method in methods:
        for split_name, split_examples_ in examples_by_split.items():
            path = pred_dir / f"{method}_{split_name}.jsonl"
            with path.open("w", encoding="utf8") as f:
                for ex in split_examples_:
                    spans = baseline_predictions[method][split_name][ex.uid]
                    payload = {"example_uid": ex.uid, "pred_spans": [[s, e] for s, e in spans]}
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_metrics_csv(rows: List[Dict[str, object]], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="outputs_qwen_mix_final")
    parser.add_argument("--methods", default="tool_overlap,lettucedetect,lookback_lens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--tune_ratio", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, or leave empty for auto/default")
    parser.add_argument(
        "--lettuce_device",
        default=None,
        help="Optional override device for LettuceDetect (e.g., cuda, cpu). Defaults to --device.",
    )
    parser.add_argument(
        "--lookback_device",
        default=None,
        help="Optional override device for LookBack Lens (e.g., cuda, cpu). Defaults to --device.",
    )
    parser.add_argument("--lettuce_model", default=DEFAULT_LETTUCE_MODEL)
    parser.add_argument("--lookback_model", default=DEFAULT_LOOKBACK_MODEL)
    parser.add_argument("--lookback_classifier_dir", default=None)
    parser.add_argument("--lookback_sliding_window", type=int, default=DEFAULT_SLIDING_WINDOW)
    parser.add_argument("--lookback_threshold", type=float, default=0.5)
    parser.add_argument(
        "--lookback_max_length",
        type=int,
        default=2048,
        help="Maximum token length for LookBack Lens prompt+answer encoding.",
    )
    parser.add_argument("--output_dir", default=None, help="Default: outputs/ensemble_voting/run_<timestamp>_seed<seed>")
    parser.add_argument("--limit_per_file", type=int, default=None, help="Use first N rows per file for smoke tests.")
    parser.add_argument(
        "--weight_values",
        default="1,2",
        help="Comma-separated integer grid for method weights, e.g. 1,2 or 1,2,3",
    )
    parser.add_argument(
        "--threshold_values",
        default=None,
        help="Optional comma-separated thresholds. If omitted, per-combination thresholds are 1..sum(weights).",
    )
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    weight_values = parse_number_list(args.weight_values, as_int=True)
    threshold_values = None if args.threshold_values is None else parse_number_list(args.threshold_values, as_int=False)
    output_dir = resolve_output_dir(args.output_dir, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Methods: {methods}")
    print(f"Dataset dir: {args.dataset_dir}")

    examples, n_rows = load_aligned_dataset(args.dataset_dir, limit_per_file=args.limit_per_file)
    split_indices = build_splits(n_rows=n_rows, seed=args.seed, test_ratio=args.test_ratio, tune_ratio=args.tune_ratio)
    examples_by_split = split_examples(examples, split_indices)

    print(
        "Split sizes by aligned row index: "
        f"train={len(split_indices['train'])}, tune={len(split_indices['tune'])}, test={len(split_indices['test'])}"
    )
    print(
        "Example counts: "
        f"train={len(examples_by_split['train'])}, "
        f"tune={len(examples_by_split['tune'])}, "
        f"test={len(examples_by_split['test'])}"
    )

    lookback_classifier_dir = (
        Path(args.lookback_classifier_dir).resolve()
        if args.lookback_classifier_dir
        else (output_dir / "lookback_classifier").resolve()
    )
    if "lookback_lens" in methods:
        print("Training LookBack Lens classifier on non-test examples...")
        train_lookback_on_non_test(
            non_test_examples=examples_by_split["non_test"],
            classifier_dir=lookback_classifier_dir,
            model_name=args.lookback_model,
            device=args.lookback_device or args.device,
            sliding_window=args.lookback_sliding_window,
            threshold=args.lookback_threshold,
            max_length=args.lookback_max_length,
        )
        print(f"Saved LookBack Lens classifier to: {lookback_classifier_dir}")

    baseline_predictions: Dict[str, Dict[str, Dict[str, List[Tuple[int, int]]]]] = {}
    for method in methods:
        method_device = args.device
        if method == "lettucedetect":
            method_device = args.lettuce_device or args.device
        elif method == "lookback_lens":
            method_device = args.lookback_device or args.device
        print(f"Generating baseline predictions for method={method} ...")
        baseline_predictions[method] = generate_predictions(
            method=method,
            examples_by_split=examples_by_split,
            lettuce_model=args.lettuce_model,
            device=method_device,
            lookback_classifier_dir=str(lookback_classifier_dir),
            lookback_model=args.lookback_model,
            lookback_sliding_window=args.lookback_sliding_window,
            lookback_threshold=args.lookback_threshold,
        )

    print("Running consistency checks (prediction counts, span roundtrip, metric parity)...")
    check_prediction_counts(methods, examples_by_split, baseline_predictions)
    method_masks: Dict[str, Dict[str, np.ndarray]] = {}
    for method in methods:
        all_preds_for_method: Dict[str, List[Tuple[int, int]]] = {}
        for split_name in ("train", "tune", "test", "non_test"):
            all_preds_for_method.update(baseline_predictions[method][split_name])
        method_masks[method] = build_masks_for_method_predictions(examples, all_preds_for_method)
    check_mask_roundtrip(method_masks)

    # Metric parity check against eval_baselines.py using tool_overlap on tune split.
    tool_overlap_preds_tune: Dict[str, List[Tuple[int, int]]]
    if "tool_overlap" in methods:
        tool_overlap_preds_tune = baseline_predictions["tool_overlap"]["tune"]
    else:
        on_the_fly = generate_predictions(
            method="tool_overlap",
            examples_by_split={"tune": examples_by_split["tune"]},
            lettuce_model=args.lettuce_model,
            device=args.device,
            lookback_classifier_dir=str(lookback_classifier_dir),
            lookback_model=args.lookback_model,
            lookback_sliding_window=args.lookback_sliding_window,
            lookback_threshold=args.lookback_threshold,
        )
        tool_overlap_preds_tune = on_the_fly["tune"]
    check_metric_parity_with_eval_baselines(
        examples_by_split["tune"],
        tool_overlap_preds_tune,
        workspace_tmp_dir=output_dir / "_tmp",
    )

    print("Searching ensemble weights/threshold on tune split...")
    tuned = run_ensemble_grid_search(
        tune_examples=examples_by_split["tune"],
        method_masks=method_masks,
        methods=methods,
        weight_values=[int(v) for v in weight_values],
        threshold_values=threshold_values,
    )

    print(f"Best ensemble weights: {tuned['method_weights']}")
    print(f"Best ensemble threshold: {tuned['threshold']}")
    print(f"Tune F1: {tuned['metrics']['f1']:.4f}")

    ensemble_test_predictions = ensemble_predict_for_examples(
        examples=examples_by_split["test"],
        method_masks=method_masks,
        method_weights=tuned["method_weights"],
        threshold=float(tuned["threshold"]),
    )

    overall_rows: List[Dict[str, object]] = []
    per_type_rows: List[Dict[str, object]] = []
    metrics_payload: Dict[str, object] = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "methods": methods,
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "tune_ratio": args.tune_ratio,
        "weight_values": [int(v) for v in weight_values],
        "threshold_values": threshold_values,
        "split_row_counts": {k: len(v) for k, v in split_indices.items()},
        "split_example_counts": {k: len(v) for k, v in examples_by_split.items()},
        "ensemble_tuning": tuned,
        "test_metrics": {},
        "test_metrics_by_type": {},
    }

    print("Evaluating baselines and ensemble on test split...")
    for method in methods:
        method_preds = baseline_predictions[method]["test"]
        method_metrics = evaluate_from_predictions(examples_by_split["test"], method_preds)
        method_by_type = evaluate_per_type(examples_by_split["test"], method_preds)
        metrics_payload["test_metrics"][method] = method_metrics
        metrics_payload["test_metrics_by_type"][method] = method_by_type

        overall_rows.append({"model": method, **method_metrics})
        for ctype, vals in method_by_type.items():
            per_type_rows.append({"model": method, "corruption_type": ctype, **vals})

    ensemble_metrics = evaluate_from_predictions(examples_by_split["test"], ensemble_test_predictions)
    ensemble_by_type = evaluate_per_type(examples_by_split["test"], ensemble_test_predictions)
    metrics_payload["test_metrics"]["ensemble_weighted_vote"] = ensemble_metrics
    metrics_payload["test_metrics_by_type"]["ensemble_weighted_vote"] = ensemble_by_type

    overall_rows.append({"model": "ensemble_weighted_vote", **ensemble_metrics})
    for ctype, vals in ensemble_by_type.items():
        per_type_rows.append({"model": "ensemble_weighted_vote", "corruption_type": ctype, **vals})

    comparison_rows = []
    for method in methods:
        base = metrics_payload["test_metrics"][method]
        delta_f1 = ensemble_metrics["f1"] - base["f1"]
        delta_p = ensemble_metrics["precision"] - base["precision"]
        delta_r = ensemble_metrics["recall"] - base["recall"]
        comparison_rows.append(
            {
                "baseline": method,
                "ensemble": "ensemble_weighted_vote",
                "baseline_f1": base["f1"],
                "ensemble_f1": ensemble_metrics["f1"],
                "delta_f1": delta_f1,
                "baseline_precision": base["precision"],
                "ensemble_precision": ensemble_metrics["precision"],
                "delta_precision": delta_p,
                "baseline_recall": base["recall"],
                "ensemble_recall": ensemble_metrics["recall"],
                "delta_recall": delta_r,
            }
        )
    metrics_payload["comparison"] = comparison_rows

    save_json(
        {
            "train_indices": sorted(split_indices["train"]),
            "tune_indices": sorted(split_indices["tune"]),
            "test_indices": sorted(split_indices["test"]),
        },
        output_dir / "split_indices.json",
    )
    save_prediction_artifacts(output_dir, methods, examples_by_split, baseline_predictions)

    # Save ensemble test predictions too.
    ensemble_pred_path = output_dir / "predictions" / "ensemble_weighted_vote_test.jsonl"
    with ensemble_pred_path.open("w", encoding="utf8") as f:
        for ex in examples_by_split["test"]:
            spans = ensemble_test_predictions[ex.uid]
            f.write(json.dumps({"example_uid": ex.uid, "pred_spans": [[s, e] for s, e in spans]}, ensure_ascii=False) + "\n")

    save_json(metrics_payload, output_dir / "metrics_summary.json")
    save_metrics_csv(overall_rows, output_dir / "metrics_overall.csv")
    save_metrics_csv(per_type_rows, output_dir / "metrics_by_type.csv")
    save_metrics_csv(comparison_rows, output_dir / "baseline_vs_ensemble_comparison.csv")

    print("\nTest Metrics (overall)")
    for row in overall_rows:
        print(
            f"{row['model']:>24} | TP={row['tp']:<5} FP={row['fp']:<5} FN={row['fn']:<5} "
            f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f}"
        )

    print(f"\nArtifacts saved to: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
