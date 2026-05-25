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
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional


MIN_FACT_VALUE_LEN = 2
OPENAI_RESPONSES_URL = 'https://api.openai.com/v1/responses'


def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            items.append(json.loads(s))
    return items


def normalize_local_record(record: dict) -> dict:
    if 'context' in record and 'output' in record:
        tool_output = stringify_value(record.get('context'))
        parsed = safe_json_loads(tool_output)
        return {
            'query': stringify_value(record.get('query')),
            'tool_output': tool_output,
            'model_response': stringify_value(record.get('source_output') or record.get('output')),
            'tool_names': extract_tool_names(parsed, tool_output),
        }
    return record


def write_jsonl(items: List[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def first_json_object(text: str) -> Optional[dict]:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text or ''):
        if ch != '{':
            continue
        try:
            value, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


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
    elif any(word in lower_key for word in ('author', 'creator', 'person', 'director')):
        choices = ['Alex Morgan', 'Taylor Reed', 'Jordan Blake', 'Casey Brooks']
    elif any(word in lower_key for word in ('name', 'title')):
        choices = ['Global Market Index', 'Regional Summary', 'Updated Result', 'Reference Entry']
    elif 'date' in lower_key or re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        return re.sub(r'\d{4}', str(random.choice([2023, 2024, 2026])), value, count=1)
    else:
        choices = ['unavailable', 'pending review', 'not reported', 'higher than expected']

    candidates = [choice for choice in choices if choice.lower() != lower_value]
    return random.choice(candidates) if candidates else value + ' updated'


def is_good_contradiction_fact(fact: Dict[str, str]) -> bool:
    key = fact.get('key', '').lower()
    value = fact.get('value', '').strip()
    if not value:
        return False
    if any(token in key for token in ('description', 'summary', 'overview')):
        return False
    if key.endswith('text') and len(value.split()) > 8:
        return False
    if len(value) > 80:
        return False
    return True


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


def insert_labeled_span(base: str, span_text: str, label_type: str) -> Dict[str, Any]:
    if not base:
        return append_labeled_span(base, span_text, label_type)
    sentence_end = re.search(r'(?<=[.!?])\s+', base)
    if sentence_end and sentence_end.end() < len(base):
        insert_at = sentence_end.start()
        output = f'{base[:insert_at]} {span_text}{base[insert_at:]}'
    else:
        result = append_labeled_span(base, span_text, label_type)
        return result
    start = output.find(span_text)
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


def validate_generated_span(output: str, label_text: str, label_type: str) -> Optional[Dict[str, Any]]:
    if not output or not label_text:
        return None
    start = output.find(label_text)
    if start < 0:
        return None
    return {
        'output': output,
        'hallucination_labels': [{
            'start': start,
            'end': start + len(label_text),
            'label': 'hallucination',
            'type': label_type,
            'text': label_text,
        }],
    }


def extract_response_text(response: dict) -> str:
    if response.get('output_text'):
        return response['output_text']
    chunks = []
    for item in response.get('output', []):
        for content in item.get('content', []):
            if content.get('type') in {'output_text', 'text'} and content.get('text'):
                chunks.append(content['text'])
    return ''.join(chunks)


def call_openai_json(prompt: str, model: str, api_key: str, temperature: float) -> dict:
    schema = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'output': {'type': 'string'},
            'label_text': {'type': 'string'},
            'strategy': {'type': 'string'},
        },
        'required': ['output', 'label_text', 'strategy'],
    }
    payload = {
        'model': model,
        'input': [
            {
                'role': 'system',
                'content': (
                    'You create hard span-level hallucination examples for tool-using dialogues. '
                    'Return only valid JSON matching the requested schema. The label_text must be '
                    'an exact contiguous substring of output.'
                ),
            },
            {'role': 'user', 'content': prompt},
        ],
        'temperature': temperature,
        'text': {
            'format': {
                'type': 'json_schema',
                'name': 'hallucination_corruption',
                'strict': True,
                'schema': schema,
            }
        },
    }
    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(payload).encode('utf8'),
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode('utf8')
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf8', errors='replace')
        raise RuntimeError(f'OpenAI API request failed: HTTP {exc.code}: {detail}') from exc
    data = json.loads(body)
    text = extract_response_text(data)
    parsed = first_json_object(text)
    if not parsed:
        raise RuntimeError(f'OpenAI API returned non-JSON text: {text[:300]}')
    return parsed


def llm_prompt(example: dict, corruption_type: str) -> str:
    facts = collect_facts(example.get('tool_output', ''))[:20]
    instructions = {
        'contradiction': (
            'Create a subtle contradiction by changing one fact in the assistant answer that is grounded in the tool output. '
            'Prefer an in-place edit of a number, date, status, entity, or categorical value. Do not append an obvious warning.'
        ),
        'overgeneration': (
            'Add one plausible detail that is not present in the tool output. Make it sound like a natural part of the answer, '
            'preferably by editing or extending an existing sentence instead of adding a repeated template.'
        ),
        'missing_tool': (
            'Add one subtle claim or recommendation that would require an additional tool action that was not performed, '
            'such as booking, buying, sending, reserving, filing, subscribing, or updating a record.'
        ),
    }
    payload = {
        'corruption_type': corruption_type,
        'task': instructions[corruption_type],
        'constraints': [
            'Keep the answer fluent and close to the original style.',
            'Change as little text as possible while making the hallucination real.',
            'Do not use phrases like "This statement is incorrect" or "independently verified".',
            'Return label_text as only the hallucinated span, not the whole answer.',
            'The label_text must appear exactly as a substring of output.',
        ],
        'query': example.get('query', ''),
        'tool_names': example.get('tool_names', []),
        'tool_output': example.get('tool_output', ''),
        'assistant_answer': example.get('model_response', ''),
        'extracted_tool_facts': facts,
        'return_json_shape': {
            'output': 'full corrupted assistant answer',
            'label_text': 'exact hallucinated substring in output',
            'strategy': 'short strategy name',
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def make_llm_corruption(example: dict, corruption_type: str, model: str, temperature: float) -> Optional[dict]:
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY is not set')
    generated = call_openai_json(llm_prompt(example, corruption_type), model, api_key, temperature)
    result = validate_generated_span(
        stringify_value(generated.get('output')),
        stringify_value(generated.get('label_text')),
        corruption_type,
    )
    if not result:
        return None
    strategy = generated.get('strategy') or 'llm_generated'
    return with_common_fields(example, result, corruption_type, f'llm_{strategy}')


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
    response_facts = [fact for fact in facts if fact['value'] in response and is_good_contradiction_fact(fact)]

    def contradiction_score(fact: Dict[str, str]):
        key = fact['key'].lower()
        value = fact['value']
        user_copied = value in query
        entity_key = any(token in key for token in ('location', 'topic', 'name', 'city', 'country'))
        numeric_value = bool(re.search(r'\d', value))
        data_key = any(token in key for token in ('value', 'change', 'price', 'population', 'weather', 'status', 'date', 'author'))
        return (user_copied, not data_key, not numeric_value, entity_key, -len(value))

    response_facts.sort(key=contradiction_score)
    fallback_facts = [fact for fact in facts if is_good_contradiction_fact(fact)]
    fact = response_facts[0] if response_facts else (fallback_facts[0] if fallback_facts else None)

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
        'finance': [
            'Analysts also expect this trend to continue next quarter.',
            'Trading volume is also above its 30-day average.',
            'The same source indicates lower volatility than last week.',
        ],
        'weather': [
            'Air quality is also excellent today.',
            'Humidity should stay comfortable through the evening.',
            'Conditions are expected to remain stable tomorrow morning.',
        ],
        'travel': [
            'Breakfast is included with the booking.',
            'The fare also includes one checked bag.',
            'Free cancellation is available until tomorrow.',
        ],
        'crypto': [
            'The address has already been backed up securely.',
            'The network fee is currently below the weekly average.',
            'The wallet has also passed a recent security check.',
        ],
        'public_data': [
            'The dataset was refreshed earlier today.',
            'The agency also reports the same trend for the previous week.',
            'The record has been cross-checked against the latest public release.',
        ],
        'general': [
            'The result is also confirmed by a second source.',
            'The latest update adds more detail to the same conclusion.',
            'The service also marks this result as high confidence.',
        ],
    }
    extra = random.choice(extras[infer_domain(example)])
    result = insert_labeled_span(example.get('model_response', ''), extra, 'overgeneration')
    return with_common_fields(example, result, 'overgeneration', 'domain_unsupported_clause')


def make_missing_tool(example: dict) -> dict:
    extras = {
        'finance': [
            'I can go ahead and place a buy order for you now.',
            'I can add this company to your portfolio watchlist.',
            'I can set an automatic price alert at this level.',
        ],
        'weather': [
            'I can also book a taxi for the best weather window.',
            'I can schedule an outdoor reminder for the clearest period.',
            'I can reserve a covered venue in case the forecast changes.',
        ],
        'travel': [
            'I have reserved the ticket for you.',
            'I can hold the room at this price.',
            'I can complete the booking with your saved payment method.',
        ],
        'crypto': [
            'I have transferred the funds to that address.',
            'I can save this address as your default withdrawal wallet.',
            'I can submit the transaction on-chain now.',
        ],
        'public_data': [
            'I have filed an official request to update the record.',
            'I can subscribe you to future updates from this agency.',
            'I can submit a correction request for this entry.',
        ],
        'general': [
            'I have completed the follow-up action for you.',
            'I can save this result to your account.',
            'I can send this result to the relevant service now.',
        ],
    }
    extra = random.choice(extras[infer_domain(example)])
    result = insert_labeled_span(example.get('model_response', ''), extra, 'missing_tool')
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


def make_corruption(example: dict, corruption_type: str, args: argparse.Namespace) -> dict:
    rule_generators = {
        'contradiction': make_contradiction,
        'overgeneration': make_overgeneration,
        'missing_tool': make_missing_tool,
    }
    if args.generation_mode in {'llm', 'hybrid'}:
        try:
            generated = make_llm_corruption(example, corruption_type, args.llm_model, args.llm_temperature)
            if generated:
                return generated
        except Exception as exc:
            if args.generation_mode == 'llm':
                raise
            print(f'LLM generation failed for {corruption_type}; falling back to rules: {exc}')

    return rule_generators[corruption_type](example)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=False, help='Path to ToolACE-style JSONL')
    parser.add_argument('--hf', help='HuggingFace dataset id (e.g., Team-ACE/ToolACE)')
    parser.add_argument('--hf_split', help='Optional split name to load from HF dataset (default: all splits)')
    parser.add_argument('--offline', action='store_true', help='Load Hugging Face datasets from the local cache only')
    parser.add_argument('--limit', type=int, help='Optional maximum number of normalized examples to use')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--seed', type=int, default=13, help='Random seed for deterministic corruptions')
    parser.add_argument(
        '--generation_mode',
        choices=['rules', 'hybrid', 'llm'],
        default='rules',
        help='rules: local deterministic corruptions; hybrid: try LLM then fallback; llm: require LLM success',
    )
    parser.add_argument('--llm_model', default=os.environ.get('OPENAI_MODEL', 'gpt-5-mini'), help='OpenAI model for LLM generation')
    parser.add_argument('--llm_temperature', type=float, default=0.7, help='Sampling temperature for LLM generation')
    args = parser.parse_args()
    random.seed(args.seed)

    # require at least one data source
    if not args.input and not args.hf:
        parser.error('Please provide either --input (local JSONL) or --hf (HuggingFace dataset id)')

    items = []
    if args.hf:
        if args.offline:
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'
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
        items = [normalize_local_record(record) for record in load_jsonl(args.input)]
        if args.limit:
            items = items[:args.limit]
    cleans, contras, overs, missings = [], [], [], []
    for ex in items:
        cleans.append(make_clean(ex))
        contras.append(make_corruption(ex, 'contradiction', args))
        overs.append(make_corruption(ex, 'overgeneration', args))
        missings.append(make_corruption(ex, 'missing_tool', args))

    write_jsonl(cleans, os.path.join(args.output_dir, 'clean.jsonl'))
    write_jsonl(contras, os.path.join(args.output_dir, 'contradiction.jsonl'))
    write_jsonl(overs, os.path.join(args.output_dir, 'overgeneration.jsonl'))
    write_jsonl(missings, os.path.join(args.output_dir, 'missing_tool.jsonl'))
    print(f'Wrote {len(cleans)} clean examples and {len(contras)} examples per corruption type to {args.output_dir}')


if __name__ == '__main__':
    main()
