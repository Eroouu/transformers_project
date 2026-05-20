import json
import os
import re
from typing import List, Tuple


def read_jsonl(path: str) -> List[dict]:
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


def token_spans(text: str):
    # returns list of (token, start, end)
    spans = []
    for m in re.finditer(r"\S+", text):
        spans.append((m.group(0), m.start(), m.end()))
    return spans


def spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def merge_adjacent(spans: List[Tuple[int, int]]):
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: s[0])
    merged = [list(spans[0])]
    for s in spans[1:]:
        if s[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s[1])
        else:
            merged.append([s[0], s[1]])
    return [(a, b) for a, b in merged]
