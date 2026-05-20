"""Evaluation harness for baseline hallucination detectors.

Includes a simple heuristic baseline `tool_overlap` that flags tokens in the model output
that are not present in the tool output (context). This script also provides a place
to add wrappers for LettuceDetect and LookBackLens.
"""
import argparse
import json
import re
from collections import Counter

try:
    from src.utils import read_jsonl, token_spans, spans_overlap, merge_adjacent
except ModuleNotFoundError:
    from utils import read_jsonl, token_spans, spans_overlap, merge_adjacent


def tool_overlap_predict(example):
    context = example.get('context', '')
    output = example.get('output', '')
    ctx_tokens = set(re.findall(r"\w+", context.lower()))
    toks = token_spans(output)
    pred_spans = []
    current = None
    for tok, s, e in toks:
        key = re.sub(r"\W+", "", tok).lower()
        if key and key not in ctx_tokens:
            if current is None:
                current = [s, e]
            else:
                current[1] = e
        else:
            if current is not None:
                pred_spans.append((current[0], current[1]))
                current = None
    if current is not None:
        pred_spans.append((current[0], current[1]))
    return pred_spans


def evaluate(dataset_path, method='tool_overlap'):
    items = read_jsonl(dataset_path)
    tp = 0
    fp = 0
    fn = 0
    for ex in items:
        gold = ex.get('hallucination_labels', [])
        gold_spans = [(g['start'], g['end']) for g in gold]
        if method == 'tool_overlap':
            pred_spans = tool_overlap_predict(ex)
        else:
            raise NotImplementedError('Only tool_overlap implemented in this scaffold')

        used = set()
        for p in pred_spans:
            matched = False
            for i, g in enumerate(gold_spans):
                if spans_overlap(p, g):
                    matched = True
                    used.add(i)
                    break
            if matched:
                tp += 1
            else:
                fp += 1
        fn += len(gold_spans) - len(used)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f'Method={method}  TP={tp} FP={fp} FN={fn} P={prec:.4f} R={rec:.4f} F1={f1:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to RAGTruth-style JSONL')
    parser.add_argument('--method', default='tool_overlap', help='Baseline method')
    args = parser.parse_args()
    evaluate(args.dataset, method=args.method)


if __name__ == '__main__':
    main()
