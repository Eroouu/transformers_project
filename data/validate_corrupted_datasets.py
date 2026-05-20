#!/usr/bin/env python3
"""Validate generated hallucination JSONL datasets.

Checks:
- required dataset files exist
- every line is valid JSON
- required fields are present and have expected types
- hallucination span offsets are valid and match the labeled text
- clean examples have no labels
- corrupted examples have at least one label of the expected type
"""
import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple


EXPECTED_FILES = {
    'clean': 'clean.jsonl',
    'contradiction': 'contradiction.jsonl',
    'overgeneration': 'overgeneration.jsonl',
    'missing_tool': 'missing_tool.jsonl',
}


def read_jsonl(path: str) -> Iterable[Tuple[int, dict]]:
    with open(path, 'r', encoding='utf8') as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_no, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_no}: invalid JSON: {exc}') from exc


def require(condition: bool, errors: List[str], message: str):
    if not condition:
        errors.append(message)


def validate_label(label: dict, output: str, expected_type: str, location: str, errors: List[str]):
    require(isinstance(label, dict), errors, f'{location}: label must be an object')
    if not isinstance(label, dict):
        return

    start = label.get('start')
    end = label.get('end')
    text = label.get('text')
    label_type = label.get('type')

    require(isinstance(start, int), errors, f'{location}: label.start must be int')
    require(isinstance(end, int), errors, f'{location}: label.end must be int')
    require(isinstance(text, str), errors, f'{location}: label.text must be string')
    require(label.get('label') == 'hallucination', errors, f'{location}: label.label must be "hallucination"')
    require(label_type == expected_type, errors, f'{location}: label.type must be "{expected_type}", got {label_type!r}')

    if isinstance(start, int) and isinstance(end, int):
        require(0 <= start < end <= len(output), errors, f'{location}: invalid span [{start}, {end}) for output length {len(output)}')
        if isinstance(text, str) and 0 <= start < end <= len(output):
            actual = output[start:end]
            require(actual == text, errors, f'{location}: span text mismatch: expected {text!r}, got {actual!r}')


def validate_record(record: dict, expected_type: str, location: str) -> List[str]:
    errors = []
    require(isinstance(record, dict), errors, f'{location}: record must be an object')
    if not isinstance(record, dict):
        return errors

    for field in ('query', 'context', 'output', 'hallucination_labels'):
        require(field in record, errors, f'{location}: missing required field {field!r}')

    query = record.get('query')
    context = record.get('context')
    output = record.get('output')
    labels = record.get('hallucination_labels')

    require(isinstance(query, str), errors, f'{location}: query must be string')
    require(isinstance(context, str), errors, f'{location}: context must be string')
    require(isinstance(output, str), errors, f'{location}: output must be string')
    require(isinstance(labels, list), errors, f'{location}: hallucination_labels must be list')

    if not isinstance(output, str) or not isinstance(labels, list):
        return errors

    if expected_type == 'clean':
        require(labels == [], errors, f'{location}: clean record must have no hallucination labels')
    else:
        require(len(labels) > 0, errors, f'{location}: corrupted record must have at least one hallucination label')
        for idx, label in enumerate(labels):
            validate_label(label, output, expected_type, f'{location}:label[{idx}]', errors)

    corruption_type = record.get('corruption_type')
    if corruption_type is not None:
        require(corruption_type == expected_type, errors, f'{location}: corruption_type must be {expected_type!r}, got {corruption_type!r}')

    source_output = record.get('source_output')
    if source_output is not None:
        require(isinstance(source_output, str), errors, f'{location}: source_output must be string')
        if expected_type == 'clean':
            require(source_output == output, errors, f'{location}: clean source_output should match output')

    return errors


def validate_file(path: str, expected_type: str, max_errors: int) -> Tuple[int, List[str]]:
    errors = []
    count = 0
    for line_no, record in read_jsonl(path):
        count += 1
        errors.extend(validate_record(record, expected_type, f'{path}:{line_no}'))
        if len(errors) >= max_errors:
            break
    if count == 0:
        errors.append(f'{path}: file contains no records')
    return count, errors


def validate_dataset_dir(dataset_dir: str, max_errors: int) -> Tuple[Dict[str, int], List[str]]:
    counts = {}
    errors = []
    for expected_type, filename in EXPECTED_FILES.items():
        path = os.path.join(dataset_dir, filename)
        if not os.path.exists(path):
            errors.append(f'{path}: missing expected dataset file')
            continue
        count, file_errors = validate_file(path, expected_type, max_errors=max_errors - len(errors))
        counts[expected_type] = count
        errors.extend(file_errors)
        if len(errors) >= max_errors:
            break

    nonzero_counts = set(counts.values())
    if len(nonzero_counts) > 1:
        errors.append(f'{dataset_dir}: dataset files have different record counts: {counts}')

    return counts, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', help='Directory containing clean/contradiction/overgeneration/missing_tool JSONL files')
    parser.add_argument('--max_errors', type=int, default=25, help='Stop after this many validation errors')
    args = parser.parse_args()

    counts, errors = validate_dataset_dir(args.dataset_dir, args.max_errors)
    if errors:
        print('Dataset validation failed.')
        print(f'Counts: {counts}')
        for error in errors[:args.max_errors]:
            print(f'- {error}')
        raise SystemExit(1)

    print('Dataset validation passed.')
    print(f'Counts: {counts}')


if __name__ == '__main__':
    main()
