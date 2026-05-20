#!/usr/bin/env python3
"""Generate span-labeled hallucination datasets from ToolACE-style data.

Outputs (written to --output_dir):
 - clean.jsonl
 - contradiction.jsonl
 - overgeneration.jsonl
 - missing_tool.jsonl

Each output line is a JSON object with the RAGTruth-style fields:
 - query: user query
 - context: tool output
 - output: model response (possibly corrupted)
 - hallucination_labels: list of {start, end, label, type, text}
"""
import argparse
import itertools
import json
import os
import random
import re
from typing import Any, Dict, Iterable, List, Optional


MIN_FACT_VALUE_LEN = 2


def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            items.append(json.loads(s))
    return items


def write_jsonl(items: List[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def extract_field(tool_output: str, field: str):
    if not tool_output:
        return None
    # build pattern safely (avoid f-string with unmatched '}' in regex)
    pattern = re.escape(field) + r"\W*[:=]\W*[\"\']?([^\"\'\},]+)"
    m = re.search(pattern, tool_output, flags=re.I)
    if m:
        return m.group(1).strip()
    return None


def safe_json_loads(text: str) -> Optional[Any]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ToolACE tool outputs are often pure JSON strings, but sample/local data may
    # include prefixes such as Weather_API:{...}. Try decoding from the first JSON
    # container if one exists.
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


def flatten_scalars(value: Any, prefix: str = '') -> Iterable[Dict[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f'{prefix}.{key}' if prefix else str(key)
            yield from flatten_scalars(child, next_prefix)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            next_prefix = f'{prefix}.{idx}' if prefix else str(idx)
            yield from flatten_scalars(child, next_prefix)
    elif value is not None:
        yield {'key': prefix, 'value': stringify_value(value)}


def regex_facts(text: str) -> List[Dict[str, str]]:
    facts = []
    quoted_pattern = r'(?<![\w-])([A-Za-z_][\w -]{0,40})\s*[:=]\s*["\']([^"\']{2,120})["\']'
    unquoted_pattern = r'(?<![\w-])([A-Za-z_][\w -]{0,40})\s*[:=]\s*([^"\'\},\]\n]{2,80})'
    for key, value in re.findall(quoted_pattern, text or ''):
        facts.append({'key': key.strip(), 'value': value.strip()})
    for key, value in re.findall(unquoted_pattern, text or ''):
        if value.strip().startswith(('{', '[')):
            continue
        facts.append({'key': key.strip(), 'value': value.strip()})
    return facts


def extract_tool_names(parsed: Any, tool_output: str) -> List[str]:
    names = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and item.get('name'):
                names.append(str(item['name']))
    elif isinstance(parsed, dict) and parsed.get('name'):
        names.append(str(parsed['name']))

    for name in re.findall(r'"name"\s*:\s*"([^"]+)"', tool_output or ''):
        names.append(name)
    for name in re.findall(r'([A-Za-z][\w ]+ API)\s*[:{]', tool_output or ''):
        names.append(name)
    return sorted(set(names))


def collect_facts(tool_output: str) -> List[Dict[str, str]]:
    parsed = safe_json_loads(tool_output)
    facts = list(flatten_scalars(parsed)) if parsed is not None else []
    facts.extend(regex_facts(tool_output))

    seen = set()
    cleaned = []
    for fact in facts:
        key = fact.get('key', '').strip()
        value = fact.get('value', '').strip()
        if len(value) < MIN_FACT_VALUE_LEN or len(value) > 120:
            continue
        if value.lower() in {'true', 'false', 'null', 'none'}:
            continue
        marker = (key.lower(), value.lower())
        if marker in seen:
            continue
        seen.add(marker)
        cleaned.append({'key': key, 'value': value})
    return cleaned


def mutate_number(text: str) -> Optional[str]:
    cleaned = text.replace(',', '')
    match = re.search(r'[-+]?\d+(?:\.\d+)?', cleaned)
    if not match:
        return None
    number = float(match.group(0))
    if number == 0:
        mutated = 1.0
    else:
        mutated = number * random.choice([0.72, 0.83, 1.18, 1.31])
    decimals = len(match.group(0).split('.')[1]) if '.' in match.group(0) else 0
    formatted = f'{mutated:,.{decimals}f}' if abs(mutated) >= 1000 else f'{mutated:.{decimals}f}'
    if '%' in text:
        formatted += '%'
    if text.strip().startswith('+') and not formatted.startswith('-'):
        formatted = '+' + formatted
    return formatted


def mutate_value(key: str, value: str) -> str:
    number = mutate_number(value)
    if number and number != value:
        return number

    lower_key = key.lower()
    lower_value = value.lower()
    if 'weather' in lower_key or lower_value in {'sunny', 'rainy', 'cloudy', 'snowy', 'windy', 'foggy', 'stormy', 'clear'}:
        choices = ['sunny', 'rainy', 'cloudy', 'snowy', 'windy', 'foggy', 'stormy', 'clear']
    elif 'country' in lower_key:
        choices = ['us', 'gb', 'ca', 'de', 'fr']
    elif 'language' in lower_key:
        choices = ['en', 'es', 'fr', 'de', 'zh']
    elif any(word in lower_key for word in ('status', 'state')):
        choices = ['pending', 'completed', 'cancelled', 'active', 'inactive']
    elif 'date' in lower_key or re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        return re.sub(r'\d{4}', str(random.choice([2023, 2024, 2026])), value, count=1)
    else:
        choices = ['unavailable', 'pending review', 'not reported', 'higher than expected']

    candidates = [choice for choice in choices if choice.lower() != lower_value]
    return random.choice(candidates) if candidates else value + ' updated'


def append_labeled_span(base: str, span_text: str, label_type: str) -> Dict[str, Any]:
    separator = ' ' if base and not base.endswith((' ', '\n')) else ''
    output = f'{base}{separator}{span_text}'
    start = output.rfind(span_text)
    return {
        'output': output,
        'hallucination_labels': [{
            'start': start,
            'end': start + len(span_text),
            'label': 'hallucination',
            'type': label_type,
            'text': span_text,
        }],
    }


def with_common_fields(example: dict, result: Dict[str, Any], corruption_type: str, strategy: str) -> dict:
    return {
        'query': example.get('query', ''),
        'context': example.get('tool_output', ''),
        'output': result['output'],
        'hallucination_labels': result['hallucination_labels'],
        'source_output': example.get('model_response', ''),
        'corruption_type': corruption_type,
        'corruption_strategy': strategy,
    }


def infer_domain(example: dict) -> str:
    text = ' '.join([
        example.get('query', ''),
        example.get('tool_output', ''),
        ' '.join(example.get('tool_names', [])),
    ]).lower()
    if any(word in text for word in ('market', 'stock', 'sec', 'filing', 'invest', 'price', 'finance')):
        return 'finance'
    if any(word in text for word in ('weather', 'temperature', 'forecast')):
        return 'weather'
    if any(word in text for word in ('hotel', 'flight', 'travel', 'ticket', 'booking', 'restaurant')):
        return 'travel'
    if any(word in text for word in ('address', 'ethereum', 'crypto', 'wallet')):
        return 'crypto'
    if any(word in text for word in ('tax', 'mobility', 'covid', 'population')):
        return 'public_data'
    return 'general'


def stringify_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


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


def normalize_hf_record(record: dict) -> List[dict]:
    """Map common HF/ToolACE records to this script's simple JSONL shape."""
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
            examples.append({
                'query': latest_user_before(conversations, idx),
                'tool_output': tool_output,
                'model_response': model_response,
                'tool_names': extract_tool_names(parsed, tool_output),
            })
        return examples

    tool_output = record.get('tool_output') or record.get('tool_response') or ''
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
                    tool_output = '; '.join(stringify_value(x) for x in value)
                else:
                    tool_output = stringify_value(value)
                break

    parsed = safe_json_loads(stringify_value(tool_output))
    return [{
        'query': record.get('query') or record.get('user_query') or record.get('prompt') or record.get('input') or '',
        'model_response': record.get('model_response') or record.get('response') or record.get('final_response') or record.get('assistant') or '',
        'tool_output': stringify_value(tool_output),
        'tool_names': extract_tool_names(parsed, stringify_value(tool_output)),
    }]


def make_contradiction(example: dict) -> dict:
    tool_output = example.get('tool_output', '')
    response = example.get('model_response', '')
    query = example.get('query', '')
    facts = collect_facts(tool_output)
    response_facts = [fact for fact in facts if fact['value'] in response]

    def contradiction_score(fact: Dict[str, str]):
        key = fact['key'].lower()
        value = fact['value']
        user_copied = value in query
        entity_key = any(token in key for token in ('location', 'topic', 'name', 'city', 'country'))
        return (user_copied, entity_key, -len(value))

    response_facts.sort(key=contradiction_score)
    fact = response_facts[0] if response_facts else (facts[0] if facts else None)

    if fact and fact['value'] in response:
        mutated = mutate_value(fact['key'], fact['value'])
        output = response.replace(fact['value'], mutated, 1)
        start = output.find(mutated)
        result = {
            'output': output,
            'hallucination_labels': [{
                'start': start,
                'end': start + len(mutated),
                'label': 'hallucination',
                'type': 'contradiction',
                'text': mutated,
            }],
        }
        return with_common_fields(example, result, 'contradiction', 'field_value_replacement')

    if fact:
        mutated = mutate_value(fact['key'], fact['value'])
        key = fact['key'].split('.')[-1].replace('_', ' ') or 'value'
        result = append_labeled_span(response, f'The reported {key} is {mutated}.', 'contradiction')
        return with_common_fields(example, result, 'contradiction', 'unsupported_field_statement')

    result = append_labeled_span(response, 'The tool result also confirms this was independently verified today.', 'contradiction')
    return with_common_fields(example, result, 'contradiction', 'generic_unsupported_statement')


def make_overgeneration(example: dict) -> dict:
    extras = {
        'finance': 'Analysts also expect this trend to continue next quarter.',
        'weather': 'Air quality is also excellent today.',
        'travel': 'Breakfast is included with the booking.',
        'crypto': 'The address has already been backed up securely.',
        'public_data': 'The dataset was refreshed earlier today.',
        'general': 'This result has also been independently verified by a second source.',
    }
    extra = extras[infer_domain(example)]
    result = append_labeled_span(example.get('model_response', ''), extra, 'overgeneration')
    return with_common_fields(example, result, 'overgeneration', 'domain_unsupported_clause')


def make_missing_tool(example: dict) -> dict:
    extras = {
        'finance': 'I can go ahead and place a buy order for you now.',
        'weather': 'I can also book a taxi for the best weather window.',
        'travel': 'I have reserved the ticket for you.',
        'crypto': 'I have transferred the funds to that address.',
        'public_data': 'I have filed an official request to update the record.',
        'general': 'I have completed the follow-up action for you.',
    }
    extra = extras[infer_domain(example)]
    result = append_labeled_span(example.get('model_response', ''), extra, 'missing_tool')
    return with_common_fields(example, result, 'missing_tool', 'unavailable_action_claim')


def make_clean(example: dict) -> dict:
    return {
        'query': example.get('query', ''),
        'context': example.get('tool_output', ''),
        'output': example.get('model_response', ''),
        'hallucination_labels': [],
        'source_output': example.get('model_response', ''),
        'corruption_type': 'clean',
        'corruption_strategy': 'none',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=False, help='Path to ToolACE-style JSONL')
    parser.add_argument('--hf', help='HuggingFace dataset id (e.g., Team-ACE/ToolACE)')
    parser.add_argument('--hf_split', help='Optional split name to load from HF dataset (default: all splits)')
    parser.add_argument('--limit', type=int, help='Optional maximum number of normalized examples to use')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--seed', type=int, default=13, help='Random seed for deterministic corruptions')
    args = parser.parse_args()
    random.seed(args.seed)

    # require at least one data source
    if not args.input and not args.hf:
        parser.error('Please provide either --input (local JSONL) or --hf (HuggingFace dataset id)')

    items = []
    if args.hf:
        try:
            from datasets import load_dataset
        except Exception as e:
            raise RuntimeError('Please install the `datasets` library to load HF datasets') from e
        ds = load_dataset(args.hf)
        # support loading a particular split or all splits
        if args.hf_split:
            if args.hf_split in ds:
                records = ds[args.hf_split]
            else:
                raise ValueError(f"Split '{args.hf_split}' not found in dataset {args.hf}. Available: {list(ds.keys())}")
        else:
            # ds may be a DatasetDict or a single Dataset
            if isinstance(ds, dict):
                records = itertools.chain.from_iterable(ds.values())
            else:
                records = ds

        # map HF records to simple examples expected by this script
        for r in records:
            items.extend(normalize_hf_record(r))
            if args.limit and len(items) >= args.limit:
                items = items[:args.limit]
                break
    else:
        items = load_jsonl(args.input)
        if args.limit:
            items = items[:args.limit]
    cleans, contras, overs, missings = [], [], [], []
    for ex in items:
        cleans.append(make_clean(ex))
        contras.append(make_contradiction(ex))
        overs.append(make_overgeneration(ex))
        missings.append(make_missing_tool(ex))

    write_jsonl(cleans, os.path.join(args.output_dir, 'clean.jsonl'))
    write_jsonl(contras, os.path.join(args.output_dir, 'contradiction.jsonl'))
    write_jsonl(overs, os.path.join(args.output_dir, 'overgeneration.jsonl'))
    write_jsonl(missings, os.path.join(args.output_dir, 'missing_tool.jsonl'))
    print(f'Wrote {len(cleans)} clean examples and {len(contras)} examples per corruption type to {args.output_dir}')


if __name__ == '__main__':
    main()
