#!/usr/bin/env python3
"""Merge multiple clean/corrupted JSONL dataset folders into one aligned directory."""

import argparse
import json
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def normalize_record(record: dict, *, dataset_source: str, file_stem: str, index: int) -> dict:
    merged = dict(record)
    merged["dataset_source"] = dataset_source

    example_id = merged.get("example_id")
    if not example_id:
        merged["example_id"] = f"{dataset_source}_{file_stem}_{index}"

    if merged.get("source_index") is None:
        merged["source_index"] = index

    return merged


def load_normalized(input_dir: Path, dataset_source: str) -> dict[str, list[dict]]:
    datasets: dict[str, list[dict]] = {}
    for filename in DATASET_FILES:
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        file_stem = path.stem
        datasets[filename] = [
            normalize_record(record, dataset_source=dataset_source, file_stem=file_stem, index=idx)
            for idx, record in enumerate(read_jsonl(path))
        ]
    return datasets


def _index_records(items: list[dict], key_field: str) -> dict[object, dict]:
    indexed: dict[object, dict] = {}
    for record in items:
        key = record[key_field]
        if key in indexed:
            raise ValueError(
                f"Duplicate {key_field}={key!r} within one file for source {record.get('dataset_source')!r}."
            )
        indexed[key] = record
    return indexed


def _choose_alignment_key(datasets: dict[str, list[dict]], dataset_source: str) -> str:
    by_example_id = [_index_records(items, "example_id") for items in datasets.values()]
    common_example_ids = set.intersection(*(set(maps.keys()) for maps in by_example_id))
    if common_example_ids:
        return "example_id"

    by_source_index = [_index_records(items, "source_index") for items in datasets.values()]
    common_source_indices = set.intersection(*(set(maps.keys()) for maps in by_source_index))
    if common_source_indices:
        return "source_index"

    counts = {name: len(items) for name, items in datasets.items()}
    raise ValueError(
        f"Could not align dataset source {dataset_source!r}: no shared example_id or source_index "
        f"across all four files. Counts: {counts}"
    )


def align_source_datasets(
    datasets: dict[str, list[dict]],
    dataset_source: str,
) -> tuple[dict[str, list[dict]], str, int]:
    """Keep only records present in all four corruption files for one source."""
    key_field = _choose_alignment_key(datasets, dataset_source)
    indexed = {name: _index_records(items, key_field) for name, items in datasets.items()}
    common_keys = set.intersection(*(set(maps.keys()) for maps in indexed.values()))
    if not common_keys:
        counts = {name: len(items) for name, items in datasets.items()}
        raise ValueError(
            f"Alignment for source {dataset_source!r} using {key_field} produced zero shared records. "
            f"Counts: {counts}"
        )

    ordered_keys = sorted(common_keys)

    aligned = {
        name: [indexed[name][key] for key in ordered_keys]
        for name in DATASET_FILES
    }
    return aligned, key_field, len(ordered_keys)


def merge_dataset_dirs(
    input_dirs: list[tuple[Path, str]],
    output_dir: Path,
    *,
    align: bool = True,
) -> dict[str, int]:
    merged: dict[str, list[dict]] = {name: [] for name in DATASET_FILES}

    for input_dir, dataset_source in input_dirs:
        datasets = load_normalized(input_dir, dataset_source)
        if align:
            before_counts = {name: len(items) for name, items in datasets.items()}
            datasets, key_field, kept = align_source_datasets(datasets, dataset_source)
            removed_parts = [
                f"{name}: -{before_counts[name] - len(datasets[name])}"
                for name in DATASET_FILES
            ]
            print(
                f"Aligned source {dataset_source!r} by {key_field}: kept {kept} "
                f"({', '.join(removed_parts)})"
            )
        for name, items in datasets.items():
            merged[name].extend(items)

    counts = {}
    for name, items in merged.items():
        write_jsonl(items, output_dir / name)
        counts[name] = len(items)

    return counts


def main():
    parser = argparse.ArgumentParser(description="Merge hallucination JSONL datasets into one folder.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input dataset directories in merge order.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Optional source labels for each input directory. Defaults to directory names.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for merged JSONL files.")
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Skip cross-file alignment and concatenate records as-is.",
    )
    args = parser.parse_args()

    if args.sources and len(args.sources) != len(args.inputs):
        raise ValueError("When provided, --sources must have the same length as --inputs.")

    input_dirs: list[tuple[Path, str]] = []
    for idx, raw_path in enumerate(args.inputs):
        path = Path(raw_path)
        source = args.sources[idx] if args.sources else path.name
        input_dirs.append((path, source))

    counts = merge_dataset_dirs(input_dirs, Path(args.output_dir), align=not args.no_align)

    print(f"Merged {len(input_dirs)} dataset folder(s) into {Path(args.output_dir).resolve()}")
    for name in DATASET_FILES:
        print(f"  {name}: {counts[name]} records")
    if len(set(counts.values())) == 1:
        print("All files are aligned to the same record count.")
    else:
        print("Warning: output files still have different record counts.")


if __name__ == "__main__":
    main()
