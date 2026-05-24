#!/usr/bin/env python3
"""Generate hallucination datasets with a local Qwen-style instruct model.

This script mirrors the output format of ``generate_hallucinations.py`` while
delegating corruption generation to a local instruction model such as
``Qwen/Qwen2.5-3B-Instruct``.

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
 - source_output: original answer
 - corruption_type: clean / contradiction / overgeneration / missing_tool
 - corruption_strategy: short name of the generation strategy used
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MIN_FACT_VALUE_LEN = 2
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_CORRUPTION_TYPES = ("contradiction", "overgeneration", "missing_tool")


def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            items.append(json.loads(s))
    return items


def write_jsonl(items: List[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def append_jsonl(item: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def safe_json_loads(text: str) -> Optional[Any]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def flatten_scalars(value: Any, prefix: str = "") -> Iterable[Dict[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_scalars(child, next_prefix)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            yield from flatten_scalars(child, next_prefix)
    elif value is not None:
        yield {"key": prefix, "value": stringify_value(value)}


def regex_facts(text: str) -> List[Dict[str, str]]:
    facts = []
    quoted_pattern = r'(?<![\w-])([A-Za-z_][\w -]{0,40})\s*[:=]\s*["\']([^"\']{2,120})["\']'
    unquoted_pattern = r'(?<![\w-])([A-Za-z_][\w -]{0,40})\s*[:=]\s*([^"\'\},\]\n]{2,80})'
    for key, value in re.findall(quoted_pattern, text or ""):
        facts.append({"key": key.strip(), "value": value.strip()})
    for key, value in re.findall(unquoted_pattern, text or ""):
        if value.strip().startswith(("{", "[")):
            continue
        facts.append({"key": key.strip(), "value": value.strip()})
    return facts


def extract_tool_names(parsed: Any, tool_output: str) -> List[str]:
    names = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]))
    elif isinstance(parsed, dict) and parsed.get("name"):
        names.append(str(parsed["name"]))

    for name in re.findall(r'"name"\s*:\s*"([^"]+)"', tool_output or ""):
        names.append(name)
    for name in re.findall(r"([A-Za-z][\w ]+ API)\s*[:{]", tool_output or ""):
        names.append(name)
    return sorted(set(names))


def stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def collect_facts(tool_output: str) -> List[Dict[str, str]]:
    parsed = safe_json_loads(tool_output)
    facts = list(flatten_scalars(parsed)) if parsed is not None else []
    facts.extend(regex_facts(tool_output))

    seen = set()
    cleaned = []
    for fact in facts:
        key = fact.get("key", "").strip()
        value = fact.get("value", "").strip()
        if len(value) < MIN_FACT_VALUE_LEN or len(value) > 120:
            continue
        if value.lower() in {"true", "false", "null", "none"}:
            continue
        marker = (key.lower(), value.lower())
        if marker in seen:
            continue
        seen.add(marker)
        cleaned.append({"key": key, "value": value})
    return cleaned


def mutate_number(text: str) -> Optional[str]:
    cleaned = text.replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    number = float(match.group(0))
    if number == 0:
        mutated = 1.0
    else:
        mutated = number * random.choice([0.72, 0.83, 1.18, 1.31])
    decimals = len(match.group(0).split(".")[1]) if "." in match.group(0) else 0
    formatted = f"{mutated:,.{decimals}f}" if abs(mutated) >= 1000 else f"{mutated:.{decimals}f}"
    if "%" in text:
        formatted += "%"
    if text.strip().startswith("+") and not formatted.startswith("-"):
        formatted = "+" + formatted
    return formatted


def mutate_value(key: str, value: str) -> str:
    number = mutate_number(value)
    if number and number != value:
        return number

    lower_key = key.lower()
    lower_value = value.lower()
    if "weather" in lower_key or lower_value in {"sunny", "rainy", "cloudy", "snowy", "windy", "foggy", "stormy", "clear"}:
        choices = ["sunny", "rainy", "cloudy", "snowy", "windy", "foggy", "stormy", "clear"]
    elif "country" in lower_key:
        choices = ["us", "gb", "ca", "de", "fr"]
    elif "language" in lower_key:
        choices = ["en", "es", "fr", "de", "zh"]
    elif any(word in lower_key for word in ("status", "state")):
        choices = ["pending", "completed", "cancelled", "active", "inactive"]
    elif "date" in lower_key or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return re.sub(r"\d{4}", str(random.choice([2023, 2024, 2026])), value, count=1)
    else:
        choices = ["unavailable", "pending review", "not reported", "higher than expected"]

    candidates = [choice for choice in choices if choice.lower() != lower_value]
    return random.choice(candidates) if candidates else value + " updated"


def shift_iso_date(value: str) -> str:
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value.strip())
    if not match:
        return value
    year, month, day = map(int, match.groups())
    year += random.choice([-1, 1, 2])
    month = min(12, max(1, month + random.choice([-2, -1, 1, 2])))
    day = min(28, max(1, day + random.choice([-4, -2, 2, 4])))
    return f"{year:04d}-{month:02d}-{day:02d}"


def perturb_numeric_string(value: str, key: str = "") -> str:
    stripped = value.strip()
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%", stripped):
        try:
            num = float(stripped[:-1])
        except ValueError:
            return value
        if num != 0 and random.random() < 0.5:
            mutated = -num
        else:
            mutated = num + random.choice([-3.4, -1.8, 1.6, 2.9])
        decimals = len(stripped[:-1].split(".")[1]) if "." in stripped[:-1] else 0
        return f"{mutated:+.{decimals}f}%" if stripped.startswith(("+", "-")) else f"{mutated:.{decimals}f}%"

    numeric = stripped.replace(",", "")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", numeric):
        return value

    try:
        num = float(numeric)
    except ValueError:
        return value

    lower_key = key.lower()
    if any(token in lower_key for token in ("latitude", "longitude", "lat", "lon")) and num != 0:
        mutated = -num
    else:
        mutated = num * random.choice([0.74, 0.86, 1.17, 1.29]) if num != 0 else 1.0

    decimals = len(numeric.split(".")[1]) if "." in numeric else 0
    if "," in stripped or abs(mutated) >= 1000:
        formatted = f"{mutated:,.{decimals}f}"
    else:
        formatted = f"{mutated:.{decimals}f}"
    if stripped.startswith("+") and not formatted.startswith("-"):
        formatted = "+" + formatted
    return formatted


def mutate_textual_value(key: str, value: str) -> str:
    lower_key = key.lower()
    lower_value = value.lower()
    if "language" in lower_key:
        options = ["Spanish", "French", "German", "Mandarin"]
    elif any(token in lower_key for token in ("author", "speaker", "artist", "name")):
        options = ["John Wooden", "Maya Angelou", "Winston Churchill", "Jane Austen"]
    elif any(token in lower_key for token in ("city", "county", "country", "state")):
        options = ["Chicago", "California", "Canada", "Berlin"]
    elif any(token in lower_key for token in ("id", "event_id", "code")):
        match = re.search(r"([A-Za-z]+)(\d+)", value)
        if match:
            prefix, suffix = match.groups()
            return f"{prefix}{int(suffix) + random.choice([111, 222, 333])}"
        return value + random.choice(["X", "2", "A"])
    elif value.startswith("http://") or value.startswith("https://"):
        return re.sub(r"/([^/]+)$", "/updated-resource", value, count=1)
    elif value.startswith("data:image/"):
        return value.replace("image", "application", 1)
    else:
        options = ["unavailable", "pending review", "not reported", "higher than expected"]

    candidates = [option for option in options if option.lower() != lower_value]
    return random.choice(candidates) if candidates else value + " updated"


def mutate_value_varied(key: str, value: str) -> str:
    value_type = contradiction_value_type(value)
    if value_type == "date":
        mutated = shift_iso_date(value)
        return mutated if mutated != value else mutate_value(key, value)
    if value_type in {"number", "percentage"}:
        mutated = perturb_numeric_string(value, key)
        return mutated if mutated != value else mutate_value(key, value)
    mutated = mutate_textual_value(key, value)
    return mutated if mutated != value else mutate_value(key, value)


def infer_domain(example: dict) -> str:
    text = " ".join(
        [
            example.get("query", ""),
            example.get("tool_output", ""),
            " ".join(example.get("tool_names", [])),
        ]
    ).lower()
    if any(word in text for word in ("market", "stock", "sec", "filing", "invest", "price", "finance")):
        return "finance"
    if any(word in text for word in ("weather", "temperature", "forecast")):
        return "weather"
    if any(word in text for word in ("hotel", "flight", "travel", "ticket", "booking", "restaurant")):
        return "travel"
    if any(word in text for word in ("address", "ethereum", "crypto", "wallet")):
        return "crypto"
    if any(word in text for word in ("tax", "mobility", "covid", "population")):
        return "public_data"
    return "general"


def latest_user_before(conversations: List[dict], index: int) -> str:
    for msg in reversed(conversations[:index]):
        if msg.get("from") == "user":
            return stringify_value(msg.get("value"))
    return ""


def next_assistant_after(conversations: List[dict], index: int) -> Optional[str]:
    for msg in conversations[index + 1 :]:
        role = msg.get("from")
        value = stringify_value(msg.get("value"))
        if role == "assistant" and value:
            return value
        if role == "user":
            return None
    return None


def normalize_hf_record(record: dict) -> List[dict]:
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        examples = []
        for idx, msg in enumerate(conversations):
            if msg.get("from") != "tool":
                continue
            model_response = next_assistant_after(conversations, idx)
            if not model_response:
                continue
            tool_output = stringify_value(msg.get("value"))
            parsed = safe_json_loads(tool_output)
            examples.append(
                {
                    "query": latest_user_before(conversations, idx),
                    "tool_output": tool_output,
                    "model_response": model_response,
                    "tool_names": extract_tool_names(parsed, tool_output),
                }
            )
        return examples

    tool_output = record.get("tool_output") or record.get("tool_response") or ""
    if not tool_output and isinstance(record.get("tool_call"), dict):
        for key in ("output", "tool_output", "result", "response"):
            if record["tool_call"].get(key):
                tool_output = record["tool_call"][key]
                break
    if not tool_output:
        for key in ("tool_outputs", "tools_output", "tools_outputs"):
            value = record.get(key)
            if value:
                if isinstance(value, list):
                    tool_output = "; ".join(stringify_value(x) for x in value)
                else:
                    tool_output = stringify_value(value)
                break

    parsed = safe_json_loads(stringify_value(tool_output))
    return [
        {
            "query": record.get("query")
            or record.get("user_query")
            or record.get("prompt")
            or record.get("input")
            or "",
            "model_response": record.get("model_response")
            or record.get("response")
            or record.get("final_response")
            or record.get("assistant")
            or "",
            "tool_output": stringify_value(tool_output),
            "tool_names": extract_tool_names(parsed, stringify_value(tool_output)),
        }
    ]


def make_clean(example: dict) -> dict:
    return {
        "query": example.get("query", ""),
        "context": example.get("tool_output", ""),
        "output": example.get("model_response", ""),
        "hallucination_labels": [],
        "source_output": example.get("model_response", ""),
        "corruption_type": "clean",
        "corruption_strategy": "none",
    }


def with_common_fields(example: dict, result: Dict[str, Any], corruption_type: str, strategy: str) -> dict:
    return {
        "query": example.get("query", ""),
        "context": example.get("tool_output", ""),
        "output": result["output"],
        "hallucination_labels": result["hallucination_labels"],
        "source_output": example.get("model_response", ""),
        "corruption_type": corruption_type,
        "corruption_strategy": strategy,
    }


def build_contradiction_hint(example: dict) -> str:
    response = example.get("model_response", "")
    facts = collect_facts(example.get("tool_output", ""))
    response_facts = [fact for fact in facts if fact["value"] in response]
    chosen = response_facts[0] if response_facts else (facts[0] if facts else None)
    if not chosen:
        return (
            "Return only a short contradictory replacement for one supported value. "
            "Keep the replacement in the same semantic type as the original value. "
            "Do not return a sentence, explanation, or commentary."
        )
    return (
        "Return only a short contradictory replacement for one supported value by editing a very short local span. "
        "The replacement must have the same semantic type as the original value. "
        "Do not add explanations, warnings, commentary, or any extra sentence. "
        f"Prefer replacing `{chosen['key']}={chosen['value']}`."
    )


def select_contradiction_target(example: dict) -> Optional[Dict[str, str]]:
    response = example.get("model_response", "")
    query = example.get("query", "")
    facts = collect_facts(example.get("tool_output", ""))
    response_facts = [fact for fact in facts if fact["value"] in response]

    def value_type(value: str) -> str:
        stripped = value.strip()
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%", stripped):
            return "percentage"
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", stripped.replace(",", "")):
            return "number"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
            return "date"
        if stripped.lower() in {"pending", "completed", "cancelled", "active", "inactive"}:
            return "status"
        return "text"

    def contradiction_score(fact: Dict[str, str]):
        key = fact["key"].lower()
        value = fact["value"]
        user_copied = value in query
        entity_key = any(token in key for token in ("location", "topic", "name", "city", "country"))
        kind = value_type(value)
        type_rank = {
            "percentage": 0,
            "number": 1,
            "date": 2,
            "status": 3,
            "text": 4,
        }[kind]
        short_rank = 0 if len(value) <= 20 else 1
        return (user_copied, entity_key, type_rank, short_rank, len(value))

    response_facts.sort(key=contradiction_score)
    return response_facts[0] if response_facts else (facts[0] if facts else None)


def build_overgeneration_hint(example: dict) -> str:
    domain = infer_domain(example)
    hints = {
        "finance": (
            "Return exactly one short extra sentence or clause that adds one new plausible market-related detail. "
            "Keep it on the same topic as the surrounding sentence. Do not repeat an existing value, metric, "
            "or title, and do not switch to an unrelated company or topic."
        ),
        "weather": (
            "Return exactly one short extra sentence or clause that adds one new plausible weather-related detail, "
            "such as humidity, air quality, wind, trend, or comfort. Keep it on the same topic and do not repeat existing facts."
        ),
        "travel": (
            "Return exactly one short extra sentence or clause that adds one new plausible travel detail, such as "
            "amenities, timing, or conditions. Keep it on the same topic and do not repeat existing facts."
        ),
        "crypto": (
            "Return exactly one short extra sentence or clause that adds one new plausible crypto-related detail, "
            "such as fees, confirmations, wallet state, or security. Keep it on the same topic and do not repeat existing facts."
        ),
        "public_data": (
            "Return exactly one short extra sentence or clause that adds one new plausible public-data detail, "
            "such as a trend, comparison, or refresh timing. Keep it on the same topic and do not repeat existing facts."
        ),
        "general": (
            "Return exactly one short extra sentence or clause that adds one new plausible factual detail. Keep it "
            "on the same topic and do not repeat existing facts."
        ),
    }
    return hints[domain]


def build_missing_tool_hint(example: dict) -> str:
    domain = infer_domain(example)
    hints = {
        "finance": (
            "Return exactly one short final sentence in first person that naturally offers or claims a financial action "
            "such as placing an order, executing a trade, transferring funds, or completing a transaction. "
            "The sentence must sound like a normal assistant offer."
        ),
        "weather": (
            "Return exactly one short final sentence in first person that naturally offers or claims a follow-up action "
            "such as booking, arranging, ordering, or scheduling something. The sentence must sound like a normal assistant offer."
        ),
        "travel": (
            "Return exactly one short final sentence in first person that naturally offers or claims a booking, reservation, "
            "purchase, or itinerary action. The sentence must sound like a normal assistant offer."
        ),
        "crypto": (
            "Return exactly one short final sentence in first person that naturally offers or claims a wallet, transfer, "
            "swap, signing, or on-chain action. The sentence must sound like a normal assistant offer."
        ),
        "public_data": (
            "Return exactly one short final sentence in first person that naturally offers or claims an official update, "
            "filing, request, submission, or record-change action. The sentence must sound like a normal assistant offer."
        ),
        "general": (
            "Return exactly one short final sentence in first person that naturally offers or claims a concrete action "
            "the assistant cannot actually perform with the available context. The sentence must sound like a normal assistant offer."
        ),
    }
    return hints[domain]


def build_generation_instruction(
    example: dict,
    corruption_type: str,
    contradiction_target: Optional[Dict[str, str]] = None,
) -> str:
    facts = collect_facts(example.get("tool_output", ""))
    fact_lines = "\n".join(f"- {fact['key']}: {fact['value']}" for fact in facts[:12]) or "- (none extracted)"
    tool_names = ", ".join(example.get("tool_names", [])) or "(unknown)"

    common_prefix = (
        "You are generating one hallucination fragment for a tool-grounded dialogue dataset.\n"
        "Return exactly one JSON object and nothing else.\n"
        "Write fluent, grammatical, professional English.\n"
        "Do not use meta words such as missing_tool, overgeneration, contradiction, hallucination, dataset, label, unsupported, tool, unavailable, or system capability.\n"
        "Do not add explanations, warnings, or commentary.\n"
        "Keep the text concise, natural, and user-facing.\n"
        "\n"
        f"User query:\n{example.get('query', '')}\n\n"
        f"Available tool names:\n{tool_names}\n\n"
        f"Tool output:\n{example.get('tool_output', '')}\n\n"
        f"Extracted supported facts:\n{fact_lines}\n\n"
        f"Original assistant answer:\n{example.get('model_response', '')}\n\n"
    )

    if corruption_type == "contradiction":
        if contradiction_target is None:
            raise ValueError("contradiction_target is required for contradiction generation")
        return (
            common_prefix
            + "Task:\n"
            + "Return a contradictory replacement for exactly one short supported value from the original answer.\n"
            + "Only generate the replacement text, not the whole answer.\n"
            + "The replacement must be short, natural, contradictory to the tool output, and of the same semantic type as the target value.\n"
            + "If the target is a percentage, return only a percentage. If the target is a number, return only a number. If the target is a date, return only a date.\n"
            + f"Target value to replace:\n{contradiction_target['value']}\n\n"
            + "Good examples:\n"
            + '- target `+0.68%` -> good output `-0.68%`\n'
            + '- target `2025-01-15` -> good output `2025-02-15`\n'
            + "Bad examples:\n"
            + '- bad output `This value is incorrect.`\n'
            + '- bad output `A misnamed and outdated index.`\n'
            + '- bad output `The index is falling.`\n\n'
            + "Return JSON with keys:\n"
            + '- "new_text": string\n'
            + '- "strategy": short snake_case string\n'
        )

    if corruption_type == "overgeneration":
        return (
            common_prefix
            + "Task:\n"
            + "Generate exactly one short extra factual sentence or clause that is plausible but not supported by the tool output.\n"
            + "Keep it topically close to the original answer. Do not introduce an unrelated topic.\n"
            + "Do not repeat or restate an existing value, percentage, title, metric, or fact already present in the original answer.\n"
            + "The text should read like one additional detail that fits naturally into the same answer.\n"
            + "The text will be inserted into the original answer by a separate algorithm.\n"
            + "Good examples:\n"
            + '- good output `Analysts expect the index to remain volatile through the week.`\n'
            + '- good output `The market has also shown stronger activity in large-cap tech stocks today.`\n'
            + "Bad examples:\n"
            + '- bad output `its percentage change of +0.90%.`\n'
            + '- bad output `By the way, Apple released a new product.`\n'
            + '- bad output `This is unsupported information.`\n\n'
            + "Return JSON with keys:\n"
            + '- "hallucinated_span": string\n'
            + '- "strategy": short snake_case string\n'
        )

    if corruption_type == "missing_tool":
        return (
            common_prefix
            + "Task:\n"
            + "Generate exactly one short final sentence where the assistant naturally offers or claims an action it cannot actually perform with the available context.\n"
            + "Write the sentence in first person, for example starting with `I can ...` or `I can go ahead and ...`.\n"
            + "Do not mention tools, context, assistant, system capability, or availability explicitly.\n"
            + "The sentence must sound like a natural assistant offer.\n"
            + "The text will be appended to the original answer by a separate algorithm.\n"
            + "Good examples:\n"
            + '- good output `I can place a buy order for you if you would like.`\n'
            + '- good output `I can reserve the tickets for you right away if you want.`\n'
            + "Bad examples:\n"
            + '- bad output `With the available context, the assistant can also provide...`\n'
            + '- bad output `This requires an unavailable tool.`\n'
            + '- bad output `Would you like to place a buy order for you if you would like?`\n\n'
            + "Return JSON with keys:\n"
            + '- "hallucinated_span": string\n'
            + '- "strategy": short snake_case string\n'
        )

    raise ValueError(f"Unsupported corruption type: {corruption_type}")


def _extract_balanced_json_substring(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _extract_json_candidate(text: str) -> Optional[str]:
    stripped = text.strip()
    if not stripped:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S | re.I)
    if fenced:
        return fenced.group(1)

    balanced = _extract_balanced_json_substring(stripped)
    if balanced:
        return balanced
    return None


def _regex_field(text: str, key: str) -> Optional[str]:
    pattern = rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"'
    match = re.search(pattern, text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return match.group(1)


def extract_json_object(text: str) -> Dict[str, Any]:
    parsed = safe_json_loads(text)
    if isinstance(parsed, dict):
        return parsed

    candidate = _extract_json_candidate(text)
    if candidate:
        parsed = safe_json_loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    fallback_text = candidate or text
    new_text = _regex_field(fallback_text, "new_text")
    rewritten_output = _regex_field(fallback_text, "rewritten_output")
    hallucinated_span = _regex_field(fallback_text, "hallucinated_span")
    strategy = _regex_field(fallback_text, "strategy")
    if new_text:
        return {
            "new_text": new_text,
            "strategy": strategy or "llm_rewrite",
        }
    if rewritten_output and hallucinated_span:
        return {
            "rewritten_output": rewritten_output,
            "hallucinated_span": hallucinated_span,
            "strategy": strategy or "llm_rewrite",
        }

    preview = normalize_whitespace(text)[:400]
    raise ValueError(f"Model output did not contain a valid JSON object. Preview: {preview}")


def build_messages(instruction: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a precise data generation assistant. Always return exactly one JSON object.",
        },
        {"role": "user", "content": instruction},
    ]


@dataclass
class LLMGenerator:
    model_name: str
    device: str
    max_input_length: int
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        device_map = "auto" if self.device == "auto" else self.device
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.model.eval()

    def generate_json(self, instruction: str) -> Dict[str, Any]:
        messages = build_messages(instruction)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": self.temperature > 0,
        }
        if generation_kwargs["do_sample"]:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return extract_json_object(text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


INVALID_META_PATTERNS = [
    r"\bhallucinat(?:ion|ed)?\b",
    r"\bmissing_tool\b",
    r"\bovergeneration\b",
    r"\bcontradiction\b",
    r"\bunsupported\b",
    r"\bunavailable\b",
    r"\bsystem capability\b",
    r"\bdataset\b",
    r"\blabel\b",
    r"\busing the unavailable tool\b",
    r"\busing the unavailable TOOL\b",
    r"\bwith the missing tool\b",
]


def contains_meta_language(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in INVALID_META_PATTERNS)


def normalize_inserted_sentence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if text and text[-1] not in ".!?":
        text += "."
    return text


def insert_before_closing_paragraph(source_output: str, inserted: str) -> tuple[str, int]:
    inserted = normalize_inserted_sentence(inserted)
    paragraphs = source_output.split("\n\n")
    if paragraphs:
        last = paragraphs[-1].strip().lower()
        closing_markers = (
            "this information",
            "let me know",
            "if you need more",
            "please enjoy",
            "feel free",
            "i hope",
        )
        if any(last.startswith(marker) for marker in closing_markers):
            prefix = "\n\n".join(paragraphs[:-1])
            suffix = paragraphs[-1]
            if prefix:
                output = prefix + "\n\n" + inserted + "\n\n" + suffix
                start = len(prefix) + 2
                return output, start
    separator = "\n\n" if source_output.strip() else ""
    output = source_output + separator + inserted
    start = len(source_output) + len(separator)
    return output, start


def append_final_sentence(source_output: str, sentence: str) -> tuple[str, int]:
    sentence = normalize_inserted_sentence(sentence)
    separator = "\n\n" if source_output.strip() else ""
    output = source_output + separator + sentence
    start = len(source_output) + len(separator)
    return output, start


def replace_first_occurrence(text: str, old: str, new: str) -> tuple[str, int]:
    start = text.find(old)
    if start < 0:
        raise ValueError(f"Target text not found in source_output: {old!r}")
    output = text[:start] + new + text[start + len(old) :]
    return output, start


def contradiction_value_type(value: str) -> str:
    stripped = value.strip()
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%", stripped):
        return "percentage"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", stripped.replace(",", "")):
        return "number"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
        return "date"
    if stripped.lower() in {"pending", "completed", "cancelled", "active", "inactive"}:
        return "status"
    return "text"


def validate_contradiction_replacement(old_text: str, new_text: str) -> None:
    if not new_text.strip():
        raise ValueError("new_text is empty")
    if new_text.strip() == old_text.strip():
        raise ValueError("new_text is identical to the original target value")
    if contains_meta_language(new_text):
        raise ValueError("new_text contains meta language")
    if len(new_text.strip()) > max(40, len(old_text) * 2):
        raise ValueError("new_text is too long for a local contradiction replacement")

    old_type = contradiction_value_type(old_text)
    new_type = contradiction_value_type(new_text)
    if old_type != new_type:
        raise ValueError(f"new_text type mismatch: expected {old_type}, got {new_type}")


def build_fallback_contradiction(example: dict, target: Optional[Dict[str, str]]) -> dict:
    source_output = example.get("model_response", "")
    if target and target["value"] in source_output:
        new_text = mutate_value_varied(target["key"], target["value"])
        rewritten_output, start = replace_first_occurrence(source_output, target["value"], new_text)
        result = {
            "output": rewritten_output,
            "hallucination_labels": [
                {
                    "start": start,
                    "end": start + len(new_text),
                    "label": "hallucination",
                    "type": "contradiction",
                    "text": new_text,
                }
            ],
        }
        return with_common_fields(example, result, "contradiction", "algorithmic_field_value_replacement")

    facts = collect_facts(example.get("tool_output", ""))
    if facts:
        fact = random.choice(facts[: min(5, len(facts))])
        field_name = fact["key"].split(".")[-1].replace("_", " ") or "value"
        field_value = mutate_value_varied(fact["key"], fact["value"])
        templates = [
            f"The reported {field_name} is {field_value}.",
            f"An updated {field_name} of {field_value} was also noted.",
            f"The latest {field_name} appears to be {field_value}.",
        ]
        new_text = random.choice(templates)
        rewritten_output, start = insert_before_closing_paragraph(source_output, new_text)
        inserted = rewritten_output[start : start + len(normalize_inserted_sentence(new_text))]
        result = {
            "output": rewritten_output,
            "hallucination_labels": [
                {
                    "start": start,
                    "end": start + len(inserted),
                    "label": "hallucination",
                    "type": "contradiction",
                    "text": inserted,
                }
            ],
        }
        return with_common_fields(example, result, "contradiction", "algorithmic_unsupported_field_statement")

    return with_common_fields(
        example,
        {
            "output": source_output + " The tool result also confirms this was independently verified today.",
            "hallucination_labels": [
                {
                    "start": len(source_output) + (0 if not source_output else 1),
                    "end": len(source_output) + (0 if not source_output else 1) + len("The tool result also confirms this was independently verified today."),
                    "label": "hallucination",
                    "type": "contradiction",
                    "text": "The tool result also confirms this was independently verified today.",
                }
            ],
        },
        "contradiction",
        "algorithmic_generic_unsupported_statement",
    )


def build_fallback_overgeneration(example: dict) -> dict:
    domain = infer_domain(example)
    query = example.get("query", "").lower()
    tool_output = example.get("tool_output", "")
    facts = collect_facts(tool_output)
    fact_values = [fact["value"] for fact in facts[:5]]
    extras = {
        "finance": [
            "Analysts expect large-cap technology shares to remain especially active this week.",
            "Momentum has also picked up in cyclical sectors during the session.",
            "Traders are also watching renewed strength in growth-focused names.",
        ],
        "weather": [
            "Humidity is also expected to stay fairly comfortable through the afternoon.",
            "Air quality is also expected to remain good today.",
            "Winds are also likely to stay fairly light for most of the day.",
        ],
        "travel": [
            "Breakfast is also included with the booking.",
            "Flexible cancellation is also available on this option.",
            "The property also offers a shuttle service to nearby transit hubs.",
        ],
        "crypto": [
            "Network activity has also stayed elevated during the session.",
            "Traders have also reported firmer demand during the latest move.",
            "On-chain activity has also remained relatively strong today.",
        ],
        "public_data": [
            "The dataset was also refreshed earlier today.",
            "A recent update also suggests the trend has stayed consistent.",
            "Nearby records also show a similar pattern in the latest release.",
        ],
        "general": [
            "Additional reports also point to a similar pattern.",
            "Recent updates also suggest the same trend is continuing.",
            "A broader review also indicates this result may be part of a larger pattern.",
        ],
    }
    if "quote" in query or "inspiration" in query:
        candidates = [
            "This quote is also often referenced in leadership workshops.",
            "It is also frequently shared in motivational newsletters.",
            "This line is also commonly used in personal development talks.",
        ]
    elif "zip code" in query or "mail-order" in query:
        candidates = [
            "The area is also known for dense commercial activity during the workweek.",
            "This zip code is also associated with a heavy daytime business population.",
            "The neighborhood also sees significant commuter traffic on weekdays.",
        ]
    elif "dog" in query or "breed" in query:
        candidates = [
            "Several of these breeds are also popular with first-time owners.",
            "Some of these breeds are also especially common in family households.",
            "A few of these breeds are also widely recommended for active owners.",
        ]
    elif "event" in query:
        candidates = [
            "Attendance is also expected to be strongest for the first event listed.",
            "Organizers are also anticipating elevated participation for these events.",
            "These events are also expected to draw strong interest from enterprise teams.",
        ]
    elif any(value for value in fact_values if value and value in example.get("model_response", "")):
        anchor = next(value for value in fact_values if value and value in example.get("model_response", ""))
        candidates = [
            f"Additional reporting also links {anchor} to a broader recent trend.",
            f"Recent commentary also suggests {anchor} is part of a larger pattern.",
        ]
    else:
        candidates = extras[domain]
    extra = random.choice(candidates)
    source_output = example.get("model_response", "")
    rewritten_output, start = insert_before_closing_paragraph(source_output, extra)
    inserted = rewritten_output[start : start + len(normalize_inserted_sentence(extra))]
    result = {
        "output": rewritten_output,
        "hallucination_labels": [
            {
                "start": start,
                "end": start + len(inserted),
                "label": "hallucination",
                "type": "overgeneration",
                "text": inserted,
            }
        ],
    }
    return with_common_fields(example, result, "overgeneration", "algorithmic_insert_extra_detail")


def build_fallback_missing_tool(example: dict) -> dict:
    domain = infer_domain(example)
    query = example.get("query", "").lower()
    extras = {
        "finance": [
            "I can place a trade for you if you would like.",
            "I can go ahead and set up a buy order for you if you want.",
            "I can execute that order for you right away if you would like.",
        ],
        "weather": [
            "I can schedule a ride for you around the best weather window if you want.",
            "I can book a car for you right away if you would like.",
            "I can arrange that trip timing for you if you want.",
        ],
        "travel": [
            "I can reserve the tickets for you right away if you want.",
            "I can book that itinerary for you if you would like.",
            "I can go ahead and confirm the reservation for you if you want.",
        ],
        "crypto": [
            "I can transfer the funds for you if you would like.",
            "I can submit that wallet transaction for you right away if you want.",
            "I can go ahead and complete the swap for you if you would like.",
        ],
        "public_data": [
            "I can submit the update request for you if you would like.",
            "I can file that change on your behalf right away if you want.",
            "I can go ahead and update the public record for you if you would like.",
        ],
        "general": [
            "I can take care of that follow-up action for you if you would like.",
            "I can go ahead and handle that next step for you if you want.",
            "I can complete that action for you right away if you would like.",
        ],
    }
    if any(token in query for token in ("image", "photo", "picture", "download")):
        candidates = [
            "I can download the image for you right away if you want.",
            "I can save that image for you if you would like.",
            "I can export that image for you immediately if you want.",
        ]
    elif any(token in query for token in ("audio", "alarm", "sound", "mp3")):
        candidates = [
            "I can save the audio file for you right away if you want.",
            "I can send that audio file to your device if you would like.",
            "I can set that audio as your alarm for you if you want.",
        ]
    elif any(token in query for token in ("password", "account")):
        candidates = [
            "I can securely store that password for you if you would like.",
            "I can go ahead and save those credentials for you if you want.",
            "I can add that password to your vault for you right away if you want.",
        ]
    elif any(token in query for token in ("quote", "quotes", "inspiration")):
        candidates = [
            "I can turn those quotes into a poster for you if you would like.",
            "I can save those quotes into a shareable card for you if you want.",
            "I can send those quotes to your notes app for you if you would like.",
        ]
    else:
        candidates = extras[domain]
    extra = random.choice(candidates)
    source_output = example.get("model_response", "")
    rewritten_output, start = append_final_sentence(source_output, extra)
    inserted = rewritten_output[start : start + len(normalize_inserted_sentence(extra))]
    result = {
        "output": rewritten_output,
        "hallucination_labels": [
            {
                "start": start,
                "end": start + len(inserted),
                "label": "hallucination",
                "type": "missing_tool",
                "text": inserted,
            }
        ],
    }
    return with_common_fields(example, result, "missing_tool", "algorithmic_unavailable_action_claim")


def generation_source(item: dict) -> str:
    strategy = item.get("corruption_strategy", "")
    strategy = str(strategy)
    if strategy.startswith(("algorithmic_", "fallback_")):
        return "algorithmic"
    return "llm"


def should_use_llm(example_idx: int, corruption_type: str, llm_fraction: float, seed: int) -> bool:
    if llm_fraction <= 0:
        return False
    if llm_fraction >= 1:
        return True
    digest = hashlib.sha256(f"{seed}:{example_idx}:{corruption_type}".encode("utf8")).digest()
    score = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return score < llm_fraction


def locate_hallucinated_span(source_output: str, rewritten_output: str, requested_span: str) -> Optional[tuple[int, int, str]]:
    span = requested_span.strip()
    if span:
        occurrences = [m.start() for m in re.finditer(re.escape(span), rewritten_output)]
        if len(occurrences) == 1:
            start = occurrences[0]
            return start, start + len(span), span

    matcher = SequenceMatcher(a=source_output, b=rewritten_output)
    fragments = []
    for tag, _, _, j1, j2 in matcher.get_opcodes():
        if tag in {"insert", "replace"} and j2 > j1:
            fragment = rewritten_output[j1:j2].strip()
            if fragment:
                fragments.append((j1, j2, fragment))
    if len(fragments) == 1:
        return fragments[0]
    return None


def validate_candidate(
    example: dict,
    corruption_type: str,
    rewritten_output: str,
    hallucinated_span: str,
    min_similarity: float,
) -> Dict[str, Any]:
    source_output = example.get("model_response", "")
    if not rewritten_output.strip():
        raise ValueError("rewritten_output is empty")
    if rewritten_output.strip() == source_output.strip():
        raise ValueError("rewritten_output is identical to source_output")

    similarity = SequenceMatcher(a=source_output, b=rewritten_output).ratio()
    if similarity < min_similarity:
        raise ValueError(
            f"rewritten_output changed too much from the source answer (similarity={similarity:.3f})"
        )

    located = locate_hallucinated_span(source_output, rewritten_output, hallucinated_span)
    if located is None:
        raise ValueError("Could not locate a unique hallucinated span in rewritten_output")
    start, end, span_text = located

    if rewritten_output[start:end] != span_text:
        raise ValueError("Span text does not match rewritten_output slice")

    if corruption_type in {"overgeneration", "missing_tool"} and span_text in source_output:
        raise ValueError("Hallucinated span already appears in source_output")

    return {
        "output": rewritten_output,
        "hallucination_labels": [
            {
                "start": start,
                "end": end,
                "label": "hallucination",
                "type": corruption_type,
                "text": span_text,
            }
        ],
    }


def generate_corruption(
    generator: Optional[LLMGenerator],
    example: dict,
    corruption_type: str,
    min_similarity: float,
    max_attempts: int,
    use_llm: bool,
) -> dict:
    contradiction_target = select_contradiction_target(example) if corruption_type == "contradiction" else None
    if not use_llm or generator is None:
        if corruption_type == "contradiction":
            return build_fallback_contradiction(example, contradiction_target)
        if corruption_type == "overgeneration":
            return build_fallback_overgeneration(example)
        if corruption_type == "missing_tool":
            return build_fallback_missing_tool(example)
        raise RuntimeError(f"Unsupported corruption type: {corruption_type}")

    last_error = None
    for attempt in range(max_attempts):
        instruction = build_generation_instruction(example, corruption_type, contradiction_target=contradiction_target)
        if attempt:
            instruction += (
                "\nPrevious attempt was rejected. Return a shorter, cleaner fragment with no meta wording.\n"
            )
        try:
            payload = generator.generate_json(instruction)
            strategy = str(payload.get("strategy", "")).strip() or "llm_rewrite"
            source_output = example.get("model_response", "")

            if corruption_type == "contradiction":
                if contradiction_target is None:
                    raise ValueError("No contradiction target could be selected")
                new_text = str(payload.get("new_text", "")).strip()
                validate_contradiction_replacement(contradiction_target["value"], new_text)
                rewritten_output, start = replace_first_occurrence(source_output, contradiction_target["value"], new_text)
                result = {
                    "output": rewritten_output,
                    "hallucination_labels": [
                        {
                            "start": start,
                            "end": start + len(new_text),
                            "label": "hallucination",
                            "type": corruption_type,
                            "text": new_text,
                        }
                    ],
                }
            elif corruption_type == "overgeneration":
                hallucinated_span = str(payload.get("hallucinated_span", "")).strip()
                if not hallucinated_span:
                    raise ValueError("hallucinated_span is empty")
                if contains_meta_language(hallucinated_span):
                    raise ValueError("hallucinated_span contains meta language")
                rewritten_output, start = insert_before_closing_paragraph(source_output, hallucinated_span)
                inserted = rewritten_output[start : start + len(normalize_inserted_sentence(hallucinated_span))]
                result = {
                    "output": rewritten_output,
                    "hallucination_labels": [
                        {
                            "start": start,
                            "end": start + len(inserted),
                            "label": "hallucination",
                            "type": corruption_type,
                            "text": inserted,
                        }
                    ],
                }
            elif corruption_type == "missing_tool":
                hallucinated_span = str(payload.get("hallucinated_span", "")).strip()
                if not hallucinated_span:
                    raise ValueError("hallucinated_span is empty")
                if contains_meta_language(hallucinated_span):
                    raise ValueError("hallucinated_span contains meta language")
                rewritten_output, start = append_final_sentence(source_output, hallucinated_span)
                inserted = rewritten_output[start : start + len(normalize_inserted_sentence(hallucinated_span))]
                result = {
                    "output": rewritten_output,
                    "hallucination_labels": [
                        {
                            "start": start,
                            "end": start + len(inserted),
                            "label": "hallucination",
                            "type": corruption_type,
                            "text": inserted,
                        }
                    ],
                }
            else:
                raise ValueError(f"Unsupported corruption type: {corruption_type}")

            result = validate_candidate(
                example=example,
                corruption_type=corruption_type,
                rewritten_output=result["output"],
                hallucinated_span=result["hallucination_labels"][0]["text"],
                min_similarity=min_similarity,
            )
            return with_common_fields(example, result, corruption_type, strategy)
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    if corruption_type == "contradiction":
        return build_fallback_contradiction(example, contradiction_target)
    if corruption_type == "overgeneration":
        return build_fallback_overgeneration(example)
    if corruption_type == "missing_tool":
        return build_fallback_missing_tool(example)

    raise RuntimeError(f"Unsupported corruption type: {corruption_type}. Last error: {last_error}")


def load_source_examples(args) -> List[dict]:
    if args.hf:
        try:
            from datasets import load_dataset
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Please install the `datasets` library to load HF datasets") from exc
        ds = load_dataset(args.hf)
        if args.hf_split:
            if args.hf_split in ds:
                records = ds[args.hf_split]
            else:
                raise ValueError(f"Split '{args.hf_split}' not found in dataset {args.hf}. Available: {list(ds.keys())}")
        else:
            records = itertools.chain.from_iterable(ds.values()) if isinstance(ds, dict) else ds

        items = []
        raw_records = 0
        for record in records:
            raw_records += 1
            items.extend(normalize_hf_record(record))
            if args.limit and len(items) >= args.limit:
                items = items[: args.limit]
                break
        print(f"Loaded {raw_records} raw HF records and normalized them into {len(items)} tool-grounded examples.")
        return items

    items = load_jsonl(args.input)
    if args.limit:
        items = items[: args.limit]
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, help="Path to ToolACE-style JSONL")
    parser.add_argument("--hf", help="HuggingFace dataset id (e.g., Team-ACE/ToolACE)")
    parser.add_argument("--hf_split", help="Optional split name to load from HF dataset (default: all splits)")
    parser.add_argument("--limit", type=int, help="Optional maximum number of normalized examples to use")
    parser.add_argument("--output_dir", default="outputs_qwen")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local causal LM checkpoint")
    parser.add_argument("--device", default="auto", help='Device or device_map, e.g. "auto", "cuda:0", "cpu"')
    parser.add_argument("--max_input_length", type=int, default=3072)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_similarity", type=float, default=0.72)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument(
        "--llm_fraction",
        type=float,
        default=0.2,
        help="Fraction of corrupted examples to generate with the LLM; the rest are generated algorithmically.",
    )
    parser.add_argument(
        "--skip_failed",
        action="store_true",
        help="Skip examples that fail LLM generation instead of aborting the full run.",
    )
    args = parser.parse_args()

    if not args.input and not args.hf:
        parser.error("Please provide either --input (local JSONL) or --hf (HuggingFace dataset id)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    items = load_source_examples(args)
    generator = None
    if args.llm_fraction > 0:
        generator = LLMGenerator(
            model_name=args.model,
            device=args.device,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )

    clean_items = []
    generated = {name: [] for name in DEFAULT_CORRUPTION_TYPES}
    source_counts = {
        name: {"llm": 0, "algorithmic": 0}
        for name in DEFAULT_CORRUPTION_TYPES
    }
    failed_log_path = os.path.join(args.output_dir, "failed_examples.jsonl")
    if args.skip_failed and os.path.exists(failed_log_path):
        os.remove(failed_log_path)

    progress = tqdm(
        items,
        desc="Generating hallucinations",
        unit="example",
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for example_idx, example in enumerate(progress):
        clean_items.append(make_clean(example))
        for corruption_type in DEFAULT_CORRUPTION_TYPES:
            try:
                generated_item = generate_corruption(
                    generator=generator,
                    example=example,
                    corruption_type=corruption_type,
                    min_similarity=args.min_similarity,
                    max_attempts=args.max_attempts,
                    use_llm=should_use_llm(example_idx, corruption_type, args.llm_fraction, args.seed),
                )
                generated[corruption_type].append(generated_item)
                source_counts[corruption_type][generation_source(generated_item)] += 1
            except Exception as exc:  # noqa: BLE001
                if not args.skip_failed:
                    raise
                append_jsonl(
                    {
                        "example_index": example_idx,
                        "corruption_type": corruption_type,
                        "error": str(exc),
                        "query": example.get("query", ""),
                        "tool_output": example.get("tool_output", ""),
                        "model_response": example.get("model_response", ""),
                    },
                    failed_log_path,
                )
        if example_idx % 5 == 0:
            progress.set_postfix(
                contradiction=len(generated["contradiction"]),
                overgeneration=len(generated["overgeneration"]),
                missing_tool=len(generated["missing_tool"]),
                c_llm=source_counts["contradiction"]["llm"],
                c_alg=source_counts["contradiction"]["algorithmic"],
                o_llm=source_counts["overgeneration"]["llm"],
                o_alg=source_counts["overgeneration"]["algorithmic"],
                m_llm=source_counts["missing_tool"]["llm"],
                m_alg=source_counts["missing_tool"]["algorithmic"],
            )

    write_jsonl(clean_items, os.path.join(args.output_dir, "clean.jsonl"))
    write_jsonl(generated["contradiction"], os.path.join(args.output_dir, "contradiction.jsonl"))
    write_jsonl(generated["overgeneration"], os.path.join(args.output_dir, "overgeneration.jsonl"))
    write_jsonl(generated["missing_tool"], os.path.join(args.output_dir, "missing_tool.jsonl"))
    print(
        f"Wrote {len(clean_items)} clean examples, "
        f"{len(generated['contradiction'])} contradiction examples, "
        f"{len(generated['overgeneration'])} overgeneration examples, and "
        f"{len(generated['missing_tool'])} missing_tool examples to {args.output_dir}"
    )
    print(
        "Generation source summary: "
        f"contradiction(llm={source_counts['contradiction']['llm']}, algorithmic={source_counts['contradiction']['algorithmic']}), "
        f"overgeneration(llm={source_counts['overgeneration']['llm']}, algorithmic={source_counts['overgeneration']['algorithmic']}), "
        f"missing_tool(llm={source_counts['missing_tool']['llm']}, algorithmic={source_counts['missing_tool']['algorithmic']})"
    )
    if args.skip_failed:
        failed_count = 0
        if os.path.exists(failed_log_path):
            with open(failed_log_path, "r", encoding="utf8") as f:
                failed_count = sum(1 for _ in f)
        print(f"Skipped {failed_count} failed corruption attempts. See {failed_log_path}.")
    print("Output format matches the original generator contract and should be compatible with validate_corrupted_datasets.py.")


if __name__ == "__main__":
    main()
