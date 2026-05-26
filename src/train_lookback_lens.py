"""Train a LookBack Lens logistic-regression classifier on local JSONL datasets."""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm

try:
    from src.lookback_lens import (
        DEFAULT_LOOKBACK_MODEL,
        DEFAULT_SLIDING_WINDOW,
        LookbackRatioExtractor,
        save_training_results,
        train_classifier,
    )
except ModuleNotFoundError:
    from lookback_lens import (
        DEFAULT_LOOKBACK_MODEL,
        DEFAULT_SLIDING_WINDOW,
        LookbackRatioExtractor,
        save_training_results,
        train_classifier,
    )


DATASET_FILES = ("clean.jsonl", "contradiction.jsonl", "overgeneration.jsonl", "missing_tool.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def resolve_dataset_paths(paths: list[str]) -> list[Path]:
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


def load_items(paths: list[Path], limit: int | None = None, seed: int = 42) -> list[dict]:
    items = []
    for path in paths:
        items.extend(read_jsonl(path))
    if limit is not None:
        random.Random(seed).shuffle(items)
        items = items[:limit]
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="One or more JSONL files or directories with clean/contradiction/overgeneration/missing_tool JSONL files.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default=DEFAULT_LOOKBACK_MODEL)
    parser.add_argument("--sliding_window", type=int, default=DEFAULT_SLIDING_WINDOW)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on loaded examples for quick runs.")
    args = parser.parse_args()

    random.seed(args.seed)
    dataset_paths = resolve_dataset_paths(args.dataset)
    items = load_items(dataset_paths, limit=args.limit, seed=args.seed)
    device = None if args.device == "auto" else args.device

    print(f"Loaded {len(items)} examples from {len(dataset_paths)} file(s).")
    print(f"Backbone model: {args.model}")
    print(f"Sliding window: {args.sliding_window}")

    extractor = LookbackRatioExtractor(
        model_name=args.model,
        device=device,
        max_length=args.max_length,
    )

    print("Extracting lookback-ratio features and training classifier...")
    bundle, training_stats = train_classifier(
        tqdm(items, desc="Training LookBack Lens", unit="example"),
        extractor=extractor,
        sliding_window=args.sliding_window,
        threshold=args.threshold,
    )
    bundle.save(args.output_dir)

    metrics = training_stats["train_window_metrics"]
    training_results = {
        "method": "lookback_lens",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": list(args.dataset),
        "output_dir": args.output_dir,
        "num_examples": len(items),
        "num_dataset_files": len(dataset_paths),
        "backbone_model": args.model,
        "sliding_window": args.sliding_window,
        "threshold": args.threshold,
        "max_length": args.max_length,
        "device": str(extractor.device),
        "seed": args.seed,
        "limit": args.limit,
        **training_stats,
    }
    results_path = save_training_results(args.output_dir, training_results)

    print(f"Saved LookBack Lens classifier to {Path(args.output_dir).resolve()}")
    print(
        "Train window metrics: "
        f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
        f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
    )
    print(f"Saved training results to {results_path.resolve()}")


if __name__ == "__main__":
    main()
