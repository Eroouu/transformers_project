#!/usr/bin/env python3
"""Build clean JSONL records from ToolACE-style data.

The normalization intentionally follows data/generate_hallucinations.py:
`context` is the original tool output, and `output` is the assistant answer
after that tool output. This script only writes clean records.
"""
import argparse
import itertools
import json
import os
import re
from typing import Any, Iterable, List, Optional


def unique_sorted(values: Iterable[Any]) -> List[str]:
    cleaned = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            cleaned.append(text)
    return sorted(set(cleaned))


def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(items: Iterable[dict], path: str) -> int:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    count = 0
    with open(path, 'w', encoding='utf8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
    return count


def stringify_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def safe_json_loads(text: str) -> Optional[Any]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in '[{':
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def extract_tool_names(parsed: Any, tool_output: str) -> List[str]:
    names = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and item.get('name'):
                names.append(str(item['name']))
    elif isinstance(parsed, dict) and parsed.get('name'):
        names.append(str(parsed['name']))

    for name in re.findall(r'([A-Za-z][\w ]+ API)\s*[:{]', tool_output or ''):
        names.append(name)
    for name in re.findall(r'([A-Za-z][\w]*_API)\s*[:{]', tool_output or ''):
        names.append(name)
    return unique_sorted(names)


def extract_available_tool_names(text: str) -> List[str]:
    if not text:
        return []
    names = []
    parsed = safe_json_loads(text)
    items = parsed if isinstance(parsed, list) else [parsed]
    for item in items:
        if isinstance(item, dict) and item.get('name'):
            names.append(str(item['name']))
    for name in re.findall(r'"name"\s*:\s*"([^"]+)"', text):
        names.append(name)
    for name in re.findall(r"'name'\s*:\s*'([^']+)'", text):
        names.append(name)
    return unique_sorted(names)


def example_available_tools(example: dict) -> List[str]:
    tools = example.get('available_tools') or example.get('tools') or []
    if isinstance(tools, str):
        return extract_available_tool_names(tools) or [tools]
    if isinstance(tools, list):
        return unique_sorted(
            str(item.get('name') if isinstance(item, dict) and item.get('name') else item)
            for item in tools
        )
    return []


def system_tools_for_record(record: dict, index: Optional[int] = None) -> List[str]:
    texts = []
    if record.get('system'):
        texts.append(stringify_value(record.get('system')))

    conversations = record.get('conversations')
    if not isinstance(conversations, list):
        return extract_available_tool_names('\n'.join(texts))

    upper_bound = index if index is not None else len(conversations)
    for msg in conversations[:upper_bound]:
        if msg.get('from') in {'system', 'human_system'}:
            texts.append(stringify_value(msg.get('value')))
    return extract_available_tool_names('\n'.join(texts))


def latest_user_before(conversations: List[dict], index: int) -> str:
    for msg in reversed(conversations[:index]):
        if msg.get('from') == 'user':
            return stringify_value(msg.get('value'))
    return ''


def next_assistant_after(conversations: List[dict], index: int) -> Optional[str]:
    for msg in conversations[index + 1:]:
        role = msg.get('from')
        value = stringify_value(msg.get('value'))
        if role == 'assistant' and value:
            return value
        if role == 'user':
            return None
    return None


def normalize_toolace_record(record: dict) -> List[dict]:
    """Map HF/local ToolACE records to the shape used by generate_hallucinations.py."""
    conversations = record.get('conversations')
    if isinstance(conversations, list):
        examples = []
        for idx, msg in enumerate(conversations):
            if msg.get('from') != 'tool':
                continue
            model_response = next_assistant_after(conversations, idx)
            if not model_response:
                continue
            tool_output = stringify_value(msg.get('value'))
            parsed = safe_json_loads(tool_output)
            tool_names = extract_tool_names(parsed, tool_output)
            examples.append({
                'query': latest_user_before(conversations, idx),
                'tool_output': tool_output,
                'model_response': model_response,
                'tool_names': tool_names,
                'available_tools': system_tools_for_record(record, idx),
            })
        return examples

    raw_available_tools = (
        record.get('available_tools')
        or record.get('tools')
        or system_tools_for_record(record)
        or []
    )
    if isinstance(raw_available_tools, str):
        available_tools = extract_available_tool_names(raw_available_tools) or [raw_available_tools]
    elif isinstance(raw_available_tools, list):
        available_tools = [
            str(item.get('name') if isinstance(item, dict) and item.get('name') else item)
            for item in raw_available_tools
        ]
    else:
        available_tools = []

    tool_output = record.get('tool_output') or record.get('tool_response') or record.get('context') or ''
    if not tool_output and isinstance(record.get('tool_call'), dict):
        for key in ('output', 'tool_output', 'result', 'response'):
            if record['tool_call'].get(key):
                tool_output = record['tool_call'][key]
                break
    if not tool_output:
        for key in ('tool_outputs', 'tools_output', 'tools_outputs'):
            value = record.get(key)
            if value:
                if isinstance(value, list):
                    tool_output = '; '.join(stringify_value(item) for item in value)
                else:
                    tool_output = stringify_value(value)
                break

    tool_output = stringify_value(tool_output)
    parsed = safe_json_loads(tool_output)
    tool_names = extract_tool_names(parsed, tool_output)
    if not tool_names and isinstance(record.get('tool_call'), dict) and record['tool_call'].get('name'):
        tool_names = [str(record['tool_call']['name'])]

    return [{
        'query': record.get('query') or record.get('user_query') or record.get('prompt') or record.get('input') or '',
        'model_response': (
            record.get('model_response')
            or record.get('response')
            or record.get('final_response')
            or record.get('assistant')
            or record.get('source_output')
            or record.get('output')
            or ''
        ),
        'tool_output': tool_output,
        'tool_names': sorted(set(tool_names)),
        'available_tools': sorted(set(available_tools)),
    }]


def make_clean(example: dict, index: int, source_split: str, id_prefix: str) -> dict:
    model_response = example.get('model_response', '')
    return {
        'example_id': example.get('example_id') or f'{id_prefix}_{source_split}_{index}',
        'query': example.get('query', ''),
        'context': example.get('tool_output', ''),
        'output': model_response,
        'hallucination_labels': [],
        'available_tools': example_available_tools(example),
        'corruption_type': 'clean',
        'corruption_strategy': 'none',
        'source_index': index,
        'source_split': source_split,
    }


def iter_hf_records(dataset_id: str, split: Optional[str]) -> Iterable[tuple[str, dict]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError('Please install the `datasets` library to load Hugging Face datasets') from exc

    dataset = load_dataset(dataset_id)
    if split:
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found. Available: {list(dataset.keys())}")
        for record in dataset[split]:
            yield split, record
        return

    if isinstance(dataset, dict):
        for split_name, records in dataset.items():
            for record in records:
                yield split_name, record
    else:
        for record in dataset:
            yield 'default', record


def iter_clean_records(args: argparse.Namespace) -> Iterable[dict]:
    if args.hf:
        raw_records = iter_hf_records(args.hf, args.hf_split)
    else:
        raw_records = ((args.source_split, record) for record in load_jsonl(args.input))

    written = 0
    for _, (source_split, record) in enumerate(raw_records):
        for example in normalize_toolace_record(record):
            clean = make_clean(example, written, source_split, args.id_prefix)
            yield clean
            written += 1
            if args.limit and written >= args.limit:
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Local ToolACE-style JSONL file')
    parser.add_argument('--hf', help='Hugging Face dataset id, for example Team-ACE/ToolACE')
    parser.add_argument('--hf_split', help='Optional split name')
    parser.add_argument('--limit', type=int, help='Optional maximum number of normalized examples')
    parser.add_argument('--output', help='Output clean JSONL path')
    parser.add_argument('--output_dir', default='outputs/toolace')
    parser.add_argument('--source_split', default='local')
    parser.add_argument('--id_prefix', default='toolace')
    args = parser.parse_args()

    if not args.input and not args.hf:
        parser.error('Please provide either --input or --hf')
    if args.input and args.hf:
        parser.error('Use only one data source: --input or --hf')
    return args


def main() -> None:
    args = parse_args()
    output_path = args.output or os.path.join(args.output_dir, 'clean.jsonl')
    count = write_jsonl(iter_clean_records(args), output_path)
    print(f'Wrote {count} clean examples to {output_path}')


if __name__ == '__main__':
    main()
