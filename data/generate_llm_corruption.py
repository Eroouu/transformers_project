#!/usr/bin/env python3
"""Generate OpenAI-assisted hallucination corruptions with span labels.

The model returns only a tiny local patch. Python applies the patch and computes
character offsets locally, because LLMs are unreliable at preserving long text
and counting characters.
"""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_OPENAI_MODEL = 'gpt-4o-mini'
OPENAI_RESPONSES_ENDPOINT = 'https://api.openai.com/v1/responses'
CORRUPTION_TYPES = ('contradiction', 'overgeneration', 'missing_tool')
DEFAULT_STRATEGIES = {
    'contradiction': 'single_replacement',
    'overgeneration': 'unsupported_fact_insertion',
    'missing_tool': 'unavailable_action_insertion',
}
ACTION_VERBS = (
    'booked', 'reserved', 'purchased', 'bought', 'sold', 'transferred',
    'sent', 'submitted', 'filed', 'updated', 'cancelled', 'canceled',
    'scheduled', 'added', 'saved', 'placed', 'ordered', 'paid', 'registered',
    'confirmed', 'created',
)
BAD_PATCH_PATTERNS = (
    'mars',
    'impossible to read',
    'password123',
    'forget everything',
    'enemy',
    'hated by many',
    'bore the audience',
    'password',
    'api key',
    'access token',
    'cookie',
    'otp',
    'failurefully',
    'successlessly',
    'has been failed',
    'independently verified',
    'guaranteed',
)
BAD_OUTPUT_PATTERNS = (
    'failurefully',
    'successlessly',
    'has been failed',
    'great news! the video has been unsuccessfully downloaded',
    ' . ',
    ' , ',
    '\n.',
    '\n,',
)
SENSITIVE_CONTEXT_PATTERNS = (
    'password',
    'api key',
    'access token',
    'cookie',
    'otp',
)
OVERGENERATION_MAX_CHARS = 120
OVERGENERATION_MIN_CHARS = 20
MISSING_TOOL_MAX_CHARS = 120
MISSING_TOOL_MIN_CHARS = 20
OVERGENERATION_GENERIC_OPENINGS = (
    'recent studies show',
    'studies show that',
    'research shows that',
    'it is estimated',
    'it is estimated that',
    'interestingly',
    'additionally',
)
ASSISTANT_ACTION_PATTERNS = (
    'would you like me to book',
    'would you like me to reserve',
    'would you like me to schedule',
    'would you like me to send',
    'would you like me to purchase',
    'would you like me to buy',
    'i can book',
    'i can reserve',
    'i can schedule',
    'i can send',
    'i can purchase',
    'i can buy',
    'i booked',
    'i scheduled',
    'i sent',
    'i saved',
    'i submitted',
    'i updated',
    'i added',
    'i downloaded',
    'i converted',
    'i generated',
    'i purchased',
    'i paid',
    'i registered',
    'i confirmed',
    'i created',
    'has been successfully downloaded',
    'has been successfully converted',
    'has been successfully sent',
    'has been successfully saved',
    'has been successfully booked',
    'has been successfully scheduled',
)
MISSING_TOOL_OFFER_MARKERS = (
    'would you like me to',
    'do you want me to',
    'would you like help',
    'should i',
    'shall i',
    'i can',
    'i could',
    'let me',
)
MISSING_TOOL_ACTION_WORDS = (
    'book',
    'reserve',
    'buy',
    'purchase',
    'order',
    'pay',
    'schedule',
    'send',
    'email',
    'save',
    'download',
    'upload',
    'convert',
    'generate',
    'create',
    'register',
    'cancel',
    'update',
    'transfer',
    'file',
    'submit',
)


class RateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


def read_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                items.append(json.loads(stripped))
    return items


def load_dotenv(path: str = '.env') -> None:
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def read_existing_source_indices(path: str) -> set:
    if not os.path.exists(path):
        return set()
    indices = set()
    for item in read_jsonl(path):
        index = item.get('source_index')
        if isinstance(index, int):
            indices.add(index)
    return indices


def append_jsonl(items: Iterable[dict], path: str) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, 'a', encoding='utf8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def compact(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + '\n[TRUNCATED]'


def get_source_output(record: dict) -> str:
    return record.get('source_output') or record.get('output') or record.get('model_response') or ''


def parse_context_tools(record: dict) -> List[str]:
    context = record.get('context') or record.get('tool_output') or ''
    try:
        parsed = json.loads(context) if isinstance(context, str) else context
    except json.JSONDecodeError:
        parsed = None
    items = parsed if isinstance(parsed, list) else [parsed]
    names = []
    for item in items:
        if isinstance(item, dict) and item.get('name'):
            names.append(str(item['name']))
    return sorted(set(names))


def get_available_tools(record: dict) -> List[str]:
    tools = record.get('available_tools') or record.get('tools') or []
    if isinstance(tools, str):
        return [tools]
    if isinstance(tools, list):
        return [str(tool) for tool in tools]
    return []


def get_called_tools(record: dict) -> List[str]:
    tools = record.get('called_tools') or record.get('tool_calls') or record.get('tool_names') or []
    if isinstance(tools, str):
        return [tools]
    if isinstance(tools, list):
        extracted = []
        for tool in tools:
            if isinstance(tool, dict) and tool.get('name'):
                extracted.append(str(tool['name']))
            else:
                extracted.append(str(tool))
        if extracted:
            return sorted(set(extracted))
    return parse_context_tools(record)


def parse_retry_delay_seconds(body: str) -> Optional[float]:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        details = parsed.get('error', {}).get('details', [])
        for detail in details:
            retry_delay = detail.get('retryDelay') if isinstance(detail, dict) else None
            if isinstance(retry_delay, str):
                match = re.fullmatch(r'(\d+(?:\.\d+)?)s', retry_delay)
                if match:
                    return float(match.group(1))

    match = re.search(r'Please retry in (\d+(?:\.\d+)?)s', body)
    if match:
        return float(match.group(1))
    return None


def parse_retry_after_header(headers: Any) -> Optional[float]:
    retry_after = headers.get('Retry-After') if headers else None
    if retry_after is None:
        return None
    try:
        return float(retry_after)
    except ValueError:
        return None


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def patch_instructions(corruption_type: str, strategy: str) -> str:
    if corruption_type == 'contradiction':
        return f"""Create one contradiction using strategy: {strategy}.
Return only a replacement patch.

Hard constraints:
- original_text must appear exactly once in the clean answer.
- original_text must be copied exactly from the clean answer.
- replacement_text must be false according to the tool context only.
- The changed claim must be grounded in the tool response, not outside knowledge.

Choose the shortest unique factual span that works:
- Prefer values, dates, counts, percentages, prices, ratings, IDs, statuses, entities,
  locations, categories, or boolean/comparison words from the tool response.
- If a value repeats, include a nearby field label or words so original_text is unique.
- Do not choose passwords, OTPs, API keys, cookies, tokens, URLs, greetings, or closings.
- Do not choose vague adjectives, jokes, insults, or subjective sentiment.

replacement_text:
- same grammatical role and semantic type as original_text;
- short, plausible, and local;
- no extra unsupported details;
- no trailing punctuation unless original_text has it.

Examples:
- Bad original_text: "New York"; good: "**City:** New York"
- Bad original_text: "2020-09-09"; good: "**Published On:** 2020-09-09"
- Bad original_text: "football"; good: "manager placeholder image for football"
- Bad original_text: "78.5%"; good: "is **78.5%.**" if the bare value repeats."""

    if corruption_type == 'overgeneration':
        return f"""Create one overgeneration using strategy: {strategy}.
Return only an insertion patch.

Goal:
Insert exactly one short unsupported factual claim that sounds like an extra field
or detail from the same tool result, but is not present in the tool context.

Hard constraints:
- place must be an exact substring from the clean answer and appear exactly once.
- Insert new_text after place only.
- new_text must be unsupported by the tool context, not contradicted by rewriting existing facts.
- new_text should be specific to the entity already discussed in the answer.
- new_text must not claim that an action was performed, scheduled, booked, sent, saved, or updated.
- new_text must not mention private/sensitive data, URLs, passwords, tokens, or OTPs.
- new_text must not start with punctuation.

Choose place:
- Prefer the end of a complete sentence, bullet item, or list item.
- The insertion may be near the beginning, middle, or end of the answer, but it must read naturally after place.
- Do not choose a repeated word, repeated name, or tiny fragment as place.

new_text:
- one sentence or one short clause;
- {OVERGENERATION_MIN_CHARS}-{OVERGENERATION_MAX_CHARS} characters;
- domain-specific and concrete, not generic or encyclopedic;
- no more than one new factual claim;
- same language and tone as the clean answer.

Good overgeneration:
- an extra rank, count, date, rating, view count, trend, availability, condition, metric, or related item;
- a plausible additional field that the tool could have returned but did not.

Bad overgeneration:
- Generic openings such as "Recent studies show", "It is estimated", "Interestingly", or "Additionally";
- external world knowledge unrelated to the tool output;
- action claims such as "I booked it" or "I saved it";
- contradictions that replace existing values;
- sentence fragments that start with punctuation;
- long explanations or multiple new facts."""

    if corruption_type == 'missing_tool':
        return f"""Create one missing-tool hallucination using strategy: {strategy}.
Return only an insertion patch.

Goal:
Insert exactly one short offer or recommendation for an action that would require
a tool that is not available in available_tools.

Hard constraints:
- place must be an exact substring from the clean answer and appear exactly once.
- Insert new_text after place only.
- new_text must offer or recommend a concrete user-facing action: book, reserve,
  buy, order, pay, schedule, send, save, download, upload, convert, register,
  cancel, update, transfer, file, or submit.
- The action must require a tool that is not listed in available_tools.
- Do not claim the action was already completed.
- Do not add factual statistics, rankings, ratings, trends, or extra tool fields; that is overgeneration.
- Do not mention private/sensitive data, passwords, tokens, cookies, API keys, or OTPs.
- new_text must not start with punctuation.

Choose place:
- Prefer after a sentence where the assistant reports information, options, or a result.
- The insertion should sound like a natural next sentence.
- Do not choose a repeated word, repeated name, or tiny fragment as place.

new_text:
- one complete sentence;
- {MISSING_TOOL_MIN_CHARS}-{MISSING_TOOL_MAX_CHARS} characters;
- explicit offer/recommendation for an unavailable tool action;
- same language and tone as the clean answer.

Good missing_tool:
- "Would you like me to book a ticket?"
- "I can reserve the first option for you."
- "Would you like me to send this to your email?"

Bad missing_tool:
- Unsupported facts such as ratings, counts, dates, or rankings;
- vague help offers such as "Let me know if you need anything else";
- actions supported by available_tools;
- claims that the action was already completed;
- sentence fragments that start with punctuation."""

    raise ValueError(f'Unsupported corruption type: {corruption_type}')


def build_patch_prompt(
    record: dict,
    corruption_type: str,
    strategy: str,
    max_context_chars: int,
    max_source_chars: int,
) -> str:
    query = record.get('query', '')
    context = compact(record.get('context') or record.get('tool_output') or '', max_context_chars)
    source_output = compact(get_source_output(record), max_source_chars)
    available_tools = get_available_tools(record)

    if corruption_type == 'contradiction':
        schema_hint = '{"original_text": "...", "replacement_text": "..."}'
        patch_rules =  """- original_text must be copied exactly from the clean answer.
- original_text must appear exactly once in the clean answer.
- replacement_text must be short and must replace original_text in the same sentence.
- replacement_text must have the same grammatical role and semantic type as original_text when possible.
- replacement_text is the hallucinated span that Python will label."""
    elif corruption_type == 'overgeneration':
        schema_hint = '{"place": "...", "new_text": "..."}'
        patch_rules = f"""- place must be copied exactly from the clean answer.
- place must appear exactly once in the clean answer.
- place should be a complete sentence, bullet item, list item, or natural clause boundary.
- new_text must be inserted after place.
- new_text must be one unsupported factual claim, {OVERGENERATION_MIN_CHARS}-{OVERGENERATION_MAX_CHARS} characters.
- new_text must not claim any completed action or tool use.
- new_text must not start with punctuation.
- new_text must avoid generic openings like "Recent studies show" or "It is estimated".
- new_text is the hallucinated span that Python will label."""
    else:
        schema_hint = '{"place": "...", "new_text": "..."}'
        patch_rules = f"""- place must be copied exactly from the clean answer.
- place must appear exactly once in the clean answer.
- place should be a complete sentence, bullet item, list item, or natural clause boundary.
- new_text must be one complete sentence, {MISSING_TOOL_MIN_CHARS}-{MISSING_TOOL_MAX_CHARS} characters.
- new_text must offer or recommend an action that would require an unavailable tool.
- new_text must not be a factual statistic/rating/ranking/trend.
- new_text must not start with punctuation.
- new_text is the hallucinated span that Python will label."""

    return f"""You are creating a minimal patch for span-labeled hallucination data.

Task:
{patch_instructions(corruption_type, strategy)}

Rules:
- Return JSON only with this shape: {schema_hint}
- Do not return the full answer.
- Do not include start/end offsets.
{patch_rules}
- The copied text must be an exact byte-for-byte substring of the clean answer.
- Do not add markdown formatting such as **bold** unless it already exists in the exact copied text.
- Keep the hallucinated span short.
- The patch must be plausible and realistic, not cartoonish or absurd.
- Keep the edit localized and natural.
- Preserve the answer's language and tone.
- Do not invent private or sensitive data.

Input user query:
{query}

Tool context, which is the source of truth:
{context}

Clean assistant answer:
{source_output}

Available tools:
{json.dumps(available_tools, ensure_ascii=False)}
"""


def patch_response_schema(corruption_type: str) -> Dict[str, Any]:
    if corruption_type == 'contradiction':
        return {
            'type': 'object',
            'properties': {
                'original_text': {
                    'type': 'string',
                    'description': 'Exact text copied from the clean answer.',
                },
                'replacement_text': {
                    'type': 'string',
                    'description': 'Replacement text that creates the contradiction.',
                },
            },
            'required': ['original_text', 'replacement_text'],
            'additionalProperties': False,
        }

    return {
        'type': 'object',
        'properties': {
            'place': {
                'type': 'string',
                'description': 'Exact text copied from the clean answer. Insert new_text after it.',
            },
            'new_text': {
                'type': 'string',
                'description': 'Inserted hallucinated text.',
            },
        },
        'required': ['place', 'new_text'],
        'additionalProperties': False,
    }


def openai_response_format(corruption_type: str) -> dict:
    return {
        'format': {
            'type': 'json_schema',
            'name': 'hallucination_patch',
            'strict': True,
            'schema': patch_response_schema(corruption_type),
        },
    }


def build_openai_responses_body(
    prompt: str,
    corruption_type: str,
    model: str,
    temperature: float,
) -> dict:
    if corruption_type == 'contradiction':
        system_prompt = (
            'You generate minimal hallucination replacement patches for tool-calling dialogue data. '
            'Prefer atomic value/entity/date/status replacements over sentence-level edits. '
            'Return only original_text and replacement_text. Do not compute offsets.'
        )
    elif corruption_type == 'overgeneration':
        system_prompt = (
            'You generate minimal hallucination insertion patches for tool-calling dialogue data. '
            'Return only place and new_text. Insert one short unsupported factual claim after place. '
            'Do not claim tool actions. Do not compute offsets.'
        )
    else:
        system_prompt = (
            'You generate minimal missing-tool insertion patches for tool-calling dialogue data. '
            'Return only place and new_text. Insert one short offer or recommendation after place. '
            'The action must require an unavailable tool. '
            'Do not compute offsets.'
        )
    return {
        'model': model,
        'input': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt},
        ],
        'temperature': temperature,
        'text': openai_response_format(corruption_type),
    }


def extract_openai_response_text(parsed: dict) -> str:
    output_text = parsed.get('output_text')
    if isinstance(output_text, str):
        return output_text
    parts = []
    for item in parsed.get('output', []):
        if not isinstance(item, dict):
            continue
        for content in item.get('content', []):
            if isinstance(content, dict) and content.get('type') in ('output_text', 'text'):
                text = content.get('text')
                if isinstance(text, str):
                    parts.append(text)
    if parts:
        return ''.join(parts)
    raise RuntimeError(f'Unexpected OpenAI response: {json.dumps(parsed)[:1000]}')


def call_openai(
    prompt: str,
    corruption_type: str,
    model: str,
    api_key: str,
    temperature: float,
    timeout: int,
) -> dict:
    payload = build_openai_responses_body(
        prompt=prompt,
        corruption_type=corruption_type,
        model=model,
        temperature=temperature,
    )
    data = json.dumps(payload).encode('utf8')
    request = urllib.request.Request(
        OPENAI_RESPONSES_ENDPOINT,
        data=data,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode('utf8')
    except urllib.error.HTTPError as exc:
        body = exc.read().decode('utf8', errors='replace')
        if exc.code == 429:
            retry_after = parse_retry_after_header(exc.headers) or parse_retry_delay_seconds(body)
            raise RateLimitError(f'OpenAI HTTP 429 rate limit: retry_after={retry_after}s', retry_after) from exc
        raise RuntimeError(f'OpenAI HTTP {exc.code}: {body}') from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f'OpenAI request failed: {exc}') from exc
    except TimeoutError as exc:
        raise RuntimeError(f'OpenAI request timed out after {timeout}s') from exc

    parsed = json.loads(raw)
    return parse_json_object(extract_openai_response_text(parsed))


def exact_once(text: str, needle: str, field_name: str) -> int:
    if not isinstance(needle, str) or not needle.strip():
        raise ValueError(f'{field_name} must be a non-empty string')
    count = text.count(needle)
    if count != 1:
        raise ValueError(f'{field_name} must appear exactly once in source_output, got {count}: {needle!r}')
    return text.index(needle)


def has_action_verb(text: str) -> bool:
    lowered = text.lower()
    return any(verb in lowered for verb in ACTION_VERBS)


def has_assistant_action_claim(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in ASSISTANT_ACTION_PATTERNS)


def has_missing_tool_action_claim(text: str) -> bool:
    lowered = text.lower()
    if has_assistant_action_claim(text):
        return True
    has_offer = any(marker in lowered for marker in MISSING_TOOL_OFFER_MARKERS)
    has_action = any(action in lowered for action in MISSING_TOOL_ACTION_WORDS)
    return has_offer and has_action


def normalize_inserted_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError('new_text must be a string')
    text = text.strip()
    if not text:
        raise ValueError('new_text must be a non-empty string')
    return text


def validate_patch_text_quality(text: str, field_name: str, max_chars: int) -> None:
    if len(text) > max_chars:
        raise ValueError(f'{field_name} is too long: {len(text)} chars > {max_chars}')
    lowered = text.lower()
    for pattern in BAD_PATCH_PATTERNS:
        if pattern in lowered:
            raise ValueError(f'{field_name} contains low-quality pattern: {pattern!r}')


def validate_overgeneration_text(record: dict, source_output: str, new_text: str) -> None:
    if len(new_text) < OVERGENERATION_MIN_CHARS:
        raise ValueError(
            f'overgeneration new_text is too short: {len(new_text)} chars < {OVERGENERATION_MIN_CHARS}'
        )
    validate_patch_text_quality(new_text, 'new_text', OVERGENERATION_MAX_CHARS)
    stripped = new_text.strip()
    if stripped and stripped[0] in '.,;:':
        raise ValueError('overgeneration new_text must not start with punctuation')
    context = record.get('context') or record.get('tool_output') or ''
    lowered_new_text = new_text.lower()
    for pattern in OVERGENERATION_GENERIC_OPENINGS:
        if lowered_new_text.strip().startswith(pattern):
            raise ValueError(f'overgeneration new_text uses generic opening: {pattern!r}')
    if lowered_new_text in source_output.lower():
        raise ValueError('overgeneration new_text already appears in source_output')
    if lowered_new_text in str(context).lower():
        raise ValueError('overgeneration new_text appears in tool context')


def validate_missing_tool_text(new_text: str) -> None:
    if len(new_text) < MISSING_TOOL_MIN_CHARS:
        raise ValueError(
            f'missing_tool new_text is too short: {len(new_text)} chars < {MISSING_TOOL_MIN_CHARS}'
        )
    validate_patch_text_quality(new_text, 'new_text', MISSING_TOOL_MAX_CHARS)
    stripped = new_text.strip()
    if stripped and stripped[0] in '.,;:':
        raise ValueError('missing_tool new_text must not start with punctuation')
    if not has_missing_tool_action_claim(new_text):
        raise ValueError(f'missing_tool new_text lacks an unavailable tool action offer or claim: {new_text!r}')


def contains_sensitive_context(record: dict, source_output: str) -> bool:
    combined = ' '.join([
        str(record.get('query', '')),
        str(record.get('context') or record.get('tool_output') or ''),
        source_output,
    ]).lower()
    return any(pattern in combined for pattern in SENSITIVE_CONTEXT_PATTERNS)


def is_word_char(char: str) -> bool:
    return char.isalnum() or char == '_'


def validate_applied_patch(
    record: dict,
    source_output: str,
    output: str,
    label_start: int,
    label_end: int,
    label_text: str,
    corruption_type: str,
) -> None:
    if output[label_start:label_end] != label_text:
        raise ValueError('Internal span mismatch after applying patch')
    if '..' in output[max(0, label_start - 2):min(len(output), label_end + 2)]:
        raise ValueError('Patch created double punctuation')
    if output.count(label_text) != 1:
        raise ValueError(f'label_text must appear exactly once in output, got {output.count(label_text)}')
    if corruption_type == 'contradiction' and len(label_text) > 50:
        raise ValueError(f'contradiction label is too long: {len(label_text)} chars > 50')

    before = output[label_start - 1] if label_start > 0 else ''
    after = output[label_end] if label_end < len(output) else ''
    if label_text and (is_word_char(label_text[0]) and is_word_char(before)):
        raise ValueError('Patch starts inside a word')
    if label_text and (is_word_char(label_text[-1]) and is_word_char(after)):
        raise ValueError('Patch ends inside a word')

    lowered_output = output.lower()
    for pattern in BAD_OUTPUT_PATTERNS:
        if pattern in lowered_output:
            raise ValueError(f'output contains low-quality pattern: {pattern!r}')

    if contains_sensitive_context(record, source_output):
        raise ValueError('sensitive credential/password/OTP-like example is skipped')


def build_patch_record(
    record: dict,
    llm_result: dict,
    corruption_type: str,
    strategy: str,
    model: str,
    index: int,
) -> dict:
    source_output = get_source_output(record)
    if not source_output:
        raise ValueError('source_output is empty')

    if corruption_type == 'contradiction':
        original_text = llm_result.get('original_text')
        replacement_text = llm_result.get('replacement_text')
        if not isinstance(replacement_text, str) or not replacement_text.strip():
            raise ValueError('replacement_text must be a non-empty string')
        replacement_text = replacement_text.strip()
        if replacement_text == original_text:
            raise ValueError('replacement_text must differ from original_text')
        validate_patch_text_quality(original_text, 'original_text', 120)
        validate_patch_text_quality(replacement_text, 'replacement_text', 120)

        start = exact_once(source_output, original_text, 'original_text')
        output = source_output[:start] + replacement_text + source_output[start + len(original_text):]
        label_start = start
        label_text = replacement_text

    else:
        place = llm_result.get('place')
        new_text = normalize_inserted_text(llm_result.get('new_text'))
        place_start = exact_once(source_output, place, 'place')
        insert_at = place_start + len(place)
        separator = '' if source_output[:insert_at].endswith((' ', '\n')) else ' '

        if corruption_type == 'overgeneration' and has_assistant_action_claim(new_text):
            raise ValueError(f'overgeneration new_text looks like an assistant action claim: {new_text!r}')
        if corruption_type == 'overgeneration':
            validate_overgeneration_text(record, source_output, new_text)
        elif corruption_type == 'missing_tool':
            validate_missing_tool_text(new_text)
        else:
            validate_patch_text_quality(new_text, 'new_text', 160)

        output = source_output[:insert_at] + separator + new_text + source_output[insert_at:]
        label_start = insert_at + len(separator)
        label_text = new_text

    label_end = label_start + len(label_text)
    validate_applied_patch(
        record=record,
        source_output=source_output,
        output=output,
        label_start=label_start,
        label_end=label_end,
        label_text=label_text,
        corruption_type=corruption_type,
    )

    return {
        'example_id': record.get('example_id') or f'{corruption_type}_{index}',
        'query': record.get('query', ''),
        'context': record.get('context') or record.get('tool_output') or '',
        'output': output,
        'hallucination_labels': [{
            'start': label_start,
            'end': label_end,
            'label': 'hallucination',
            'type': corruption_type,
            'text': label_text,
        }],
        'available_tools': get_available_tools(record),
        'corruption_type': corruption_type,
        'generation_method': 'openai_patch',
        'source_index': index,
        'source_split': record.get('source_split', ''),
    }


def is_non_retryable_validation_error(exc: Exception) -> bool:
    return 'sensitive credential/password/OTP-like example is skipped' in str(exc)


def validation_retry_guidance(error_message: str) -> str:
    if 'place must appear exactly once in source_output' in error_message:
        if 'got 0' in error_message:
            return (
                '- The previous place was not an exact substring of the clean answer.\n'
                '- Copy place byte-for-byte from the clean answer, preserving punctuation and markdown.'
            )
        return (
            '- The previous place appeared multiple times in the clean answer.\n'
            '- Do not reuse that place.\n'
            '- Choose a longer exact sentence, bullet item, list item, or natural clause that appears exactly once.'
        )
    if 'overgeneration new_text looks like an assistant action claim' in error_message:
        return (
            '- The previous new_text claimed that the assistant/tool completed an action.\n'
            '- For overgeneration, add only an unsupported factual detail, not an action performed for the user.'
        )
    if 'overgeneration new_text' in error_message:
        return (
            f'- The previous new_text failed overgeneration validation.\n'
            f'- Write exactly one concrete unsupported factual claim, {OVERGENERATION_MIN_CHARS}-{OVERGENERATION_MAX_CHARS} characters.\n'
            '- Do not start with punctuation or generic openings like "Recent studies show".\n'
            '- Make it sound like an extra field/detail from the same tool result, not broad external knowledge.'
        )
    if 'missing_tool new_text' in error_message:
        return (
            f'- The previous new_text failed missing-tool validation.\n'
            f'- Write one complete sentence, {MISSING_TOOL_MIN_CHARS}-{MISSING_TOOL_MAX_CHARS} characters.\n'
            '- It must explicitly offer or recommend an action requiring an unavailable tool.\n'
            '- Use wording like "Would you like me to book...", "I can send...", or "Should I schedule...?"'
        )
    if 'must appear exactly once in source_output' in error_message:
        if 'got 0' in error_message:
            return (
                '- The previous original_text was not an exact substring of the clean answer.\n'
                '- Copy original_text byte-for-byte, preserving case, markdown, punctuation, and spaces.'
            )
        return (
            '- The previous original_text appeared multiple times in the clean answer.\n'
            '- Do not reuse that original_text.\n'
            '- Choose a longer exact substring by adding a nearby field label, list item context, or surrounding words until it is unique.\n'
            '- Keep the hallucinated label short by changing only the false value inside that unique substring when possible.'
        )
    if 'inside a word' in error_message:
        return (
            '- The previous patch replaced text inside a larger word.\n'
            '- Choose a complete word, number, date, value, or phrase; never replace only part of a word.'
        )
    if 'label_text must appear exactly once in output' in error_message:
        return (
            '- The previous replacement_text appeared more than once in the final answer.\n'
            '- Choose a different replacement_text that will be unique in the output.'
        )
    if 'too long' in error_message:
        return (
            '- The previous hallucinated span was too long.\n'
            '- Choose a smaller atomic value or phrase, ideally under 50 characters.'
        )
    if 'low-quality pattern' in error_message or 'output contains low-quality pattern' in error_message:
        return (
            '- The previous patch created a low-quality or unnatural answer.\n'
            '- Choose a realistic factual value change instead of awkward wording.'
        )
    if 'double punctuation' in error_message:
        return (
            '- The previous patch created bad punctuation.\n'
            '- Match punctuation boundaries: include punctuation only if original_text includes it.'
        )
    return (
        '- The previous patch failed local validation.\n'
        '- Return a different exact local patch that satisfies all rules.'
    )


def build_retry_prompt(base_prompt: str, llm_result: Optional[dict], exc: Exception) -> str:
    error_message = str(exc)
    invalid_patch = json.dumps(llm_result or {}, ensure_ascii=False)
    return f"""{base_prompt}

Retry feedback from the local Python validator:
The previous JSON patch was rejected.

Invalid patch:
{invalid_patch}

Validation error:
{error_message}

Fix this error on the next attempt:
{validation_retry_guidance(error_message)}

Return only a new JSON object. Do not reuse the invalid patch."""


def generate_one(
    record: dict,
    index: int,
    args: argparse.Namespace,
    api_key: str,
) -> dict:
    strategy = args.strategy or DEFAULT_STRATEGIES[args.corruption_type]
    prompt = build_patch_prompt(
        record,
        corruption_type=args.corruption_type,
        strategy=strategy,
        max_context_chars=args.max_context_chars,
        max_source_chars=args.max_source_chars,
    )
    if args.print_prompt:
        print(prompt)
        raise SystemExit(0)

    last_error = None
    current_prompt = prompt
    last_attempt = 0
    for attempt in range(1, args.retries + 1):
        last_attempt = attempt
        llm_result = None
        try:
            llm_result = call_openai(
                prompt=current_prompt,
                corruption_type=args.corruption_type,
                model=args.model,
                api_key=api_key,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            return build_patch_record(record, llm_result, args.corruption_type, strategy, args.model, index)
        except Exception as exc:
            last_error = exc
            if llm_result is not None and is_non_retryable_validation_error(exc):
                break
            if llm_result is not None:
                current_prompt = build_retry_prompt(prompt, llm_result, exc)
            if attempt < args.retries:
                retry_sleep = args.retry_sleep
                if isinstance(exc, RateLimitError) and exc.retry_after is not None:
                    retry_sleep = max(retry_sleep, exc.retry_after + args.retry_buffer)
                print(f'[{index}] attempt {attempt} failed; retrying in {retry_sleep:.1f}s: {exc}', flush=True)
                time.sleep(retry_sleep)
    raise RuntimeError(f'Failed after {last_attempt} attempt(s): {last_error}') from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input clean JSONL with query/context/output fields')
    parser.add_argument('--output', required=True, help='Output JSONL for generated corrupted records')
    parser.add_argument('--corruption_type', required=True, choices=CORRUPTION_TYPES)
    parser.add_argument('--strategy', help='Strategy name to put into the prompt and output metadata')
    parser.add_argument('--model', default=DEFAULT_OPENAI_MODEL, help='OpenAI model name')
    parser.add_argument('--api_key_env', default='OPENAI_API_KEY', help='Environment variable with OpenAI API key')
    parser.add_argument('--env_file', default='.env', help='Optional .env file to load before reading API key')
    parser.add_argument('--limit', type=int, help='Maximum number of records to process')
    parser.add_argument('--start', type=int, default=0, help='Start index in input JSONL')
    parser.add_argument('--end', type=int, help='End index in input JSONL, exclusive')
    parser.add_argument('--temperature', type=float, default=0.4)   
    parser.add_argument('--sleep', type=float, default=0.0, help='Sleep between successful API calls')
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--retry_sleep', type=float, default=2.0)
    parser.add_argument('--retry_buffer', type=float, default=2.0, help='Extra seconds added to provider retryDelay')
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--max_context_chars', type=int, default=5000)
    parser.add_argument('--max_source_chars', type=int, default=2500)
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output file instead of appending')
    parser.add_argument('--skip_existing', action='store_true', help='Skip source_index values already present in output')
    parser.add_argument('--print_prompt', action='store_true', help='Print the first prompt and exit without API call')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.print_prompt:
        raise SystemExit(f'Missing API key. Set ${args.api_key_env} first.')

    records = read_jsonl(args.input)
    if args.start < 0:
        raise SystemExit('--start must be >= 0')
    if args.end is not None and args.end < args.start:
        raise SystemExit('--end must be >= --start')
    selected = records[args.start:args.end]
    if args.limit is not None:
        selected = selected[:args.limit]

    if args.overwrite and os.path.exists(args.output):
        os.remove(args.output)
    existing_indices = read_existing_source_indices(args.output) if args.skip_existing else set()
    if existing_indices:
        print(f'Skipping {len(existing_indices)} existing source_index value(s) from {args.output}', flush=True)

    written = 0
    failures = 0
    for offset, record in enumerate(selected, start=args.start):
        if offset in existing_indices:
            print(f'[{offset}] skipped existing', flush=True)
            continue
        try:
            generated = generate_one(record, offset, args, api_key or '')
            append_jsonl([generated], args.output)
            written += 1
            print(f'[{offset}] wrote {args.corruption_type}: {len(generated["hallucination_labels"])} span(s)', flush=True)
            if args.sleep:
                time.sleep(args.sleep)
        except Exception as exc:
            failures += 1
            print(f'[{offset}] failed: {exc}', flush=True)

    print(f'Done. written={written} failures={failures} output={args.output}')


if __name__ == '__main__':
    main()
