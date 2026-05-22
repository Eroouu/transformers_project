#!/usr/bin/env python3
"""Split generated clean/corrupted JSONL datasets into aligned train/test folders."""

import argparse
import json
import os
import random
from pathlib import Path


DATASET_FILES = ("clean.jsonl", "contradiction.jsonl", "overgeneration.jsonl", "missing_tool.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(items: list[dict], path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    datasets = {name: read_jsonl(input_dir / name) for name in DATASET_FILES}
    counts = {name: len(items) for name, items in datasets.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"All dataset files must have the same number of rows. Counts: {counts}")

    total = next(iter(counts.values()))
    indices = list(range(total))
    random.Random(args.seed).shuffle(indices)
    test_size = max(1, int(total * args.test_ratio))
    test_indices = set(indices[:test_size])

    for name, items in datasets.items():
        train_items = [item for idx, item in enumerate(items) if idx not in test_indices]
        test_items = [item for idx, item in enumerate(items) if idx in test_indices]
        write_jsonl(train_items, Path(args.train_dir) / name)
        write_jsonl(test_items, Path(args.test_dir) / name)

    print(f"Split {total} aligned examples per file.")
    print(f"Train: {total - test_size} examples per file -> {args.train_dir}")
    print(f"Test: {test_size} examples per file -> {args.test_dir}")


if __name__ == "__main__":
    main()
