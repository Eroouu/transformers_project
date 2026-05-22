"""Evaluation harness for baseline hallucination detectors.

Includes a simple heuristic baseline `tool_overlap` that flags tokens in the model output
that are not present in the tool output (context), plus optional LettuceDetect and
LookBack Lens wrappers for span-level hallucination detection.
"""
import argparse
import os
import re

from tqdm.auto import tqdm

try:
    from src.utils import read_jsonl, token_spans, spans_overlap, merge_adjacent
except ModuleNotFoundError:
    from utils import read_jsonl, token_spans, spans_overlap, merge_adjacent


DEFAULT_LETTUCE_MODEL = 'KRLabsOrg/lettucedect-base-modernbert-en-v1'
DEFAULT_LOOKBACK_CLASSIFIER_DIR = 'models/lookback_lens'
DATASET_FILES = ('clean.jsonl', 'contradiction.jsonl', 'overgeneration.jsonl', 'missing_tool.jsonl')


def load_eval_items(dataset_path):
    if os.path.isdir(dataset_path):
        items = []
        loaded_files = []
        for name in DATASET_FILES:
            path = os.path.join(dataset_path, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f'Missing dataset file in directory: {path}')
            file_items = read_jsonl(path)
            items.extend(file_items)
            loaded_files.append(f'{name}={len(file_items)}')
        print(f"Loaded {len(items)} examples from {dataset_path} ({', '.join(loaded_files)})", flush=True)
        return items

    items = read_jsonl(dataset_path)
    print(f'Loaded {len(items)} examples from {dataset_path}', flush=True)
    return items


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


class LettuceDetectPredictor:
    """Thin adapter around the official lettucedetect package.

    The package returns span dictionaries for the response. This adapter normalizes
    possible key variants into the `(start, end)` tuples used by the local evaluator.
    """

    def __init__(self, model_name=DEFAULT_LETTUCE_MODEL, device=None):
        try:
            from lettucedetect.models.inference import HallucinationDetector
        except ImportError as exc:
            raise ImportError(
                "LettuceDetect is not installed. Install project requirements with "
                "`pip install -r requirements.txt`, or install it directly with "
                "`pip install lettucedetect`."
            ) from exc

        kwargs = {'method': 'transformer', 'model_path': model_name}
        if device is not None:
            kwargs['device'] = device
        self.detector = HallucinationDetector(**kwargs)

    def predict(self, example):
        context = example.get('context') or example.get('tool_output') or ''
        question = example.get('query') or ''
        answer = example.get('output') or example.get('model_response') or ''

        result = self.detector.predict(
            context=[context],
            question=question,
            answer=answer,
            output_format='spans',
        )
        return normalize_lettuce_spans(result)


class LookBackLensPredictorWrapper:
    """Adapter around the local LookBack Lens implementation."""

    def __init__(
        self,
        classifier_dir=DEFAULT_LOOKBACK_CLASSIFIER_DIR,
        lookback_model=None,
        device=None,
        sliding_window=None,
        threshold=None,
    ):
        try:
            from src.lookback_lens import LookBackLensPredictor
        except ModuleNotFoundError:
            from lookback_lens import LookBackLensPredictor

        if not os.path.isdir(classifier_dir):
            raise FileNotFoundError(
                f"LookBack Lens classifier directory not found: {classifier_dir}. "
                "Train one first with "
                "`python src/train_lookback_lens.py --dataset outputs/toolace_train "
                f"--output_dir {classifier_dir}`."
            )

        self.predictor = LookBackLensPredictor(
            classifier_dir=classifier_dir,
            model_name=lookback_model,
            device=device,
            sliding_window=sliding_window,
            threshold=threshold,
        )

    def predict(self, example):
        return self.predictor.predict(example)


def normalize_lettuce_spans(result):
    if isinstance(result, dict):
        if 'hallucinations' in result:
            result = result['hallucinations']
        elif 'spans' in result:
            result = result['spans']
        elif 'predictions' in result:
            result = result['predictions']

    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        result = result[0]

    pred_spans = []
    for span in result or []:
        if isinstance(span, (tuple, list)) and len(span) >= 2:
            pred_spans.append((int(span[0]), int(span[1])))
            continue
        if not isinstance(span, dict):
            continue

        start = span.get('start', span.get('start_char', span.get('start_index')))
        end = span.get('end', span.get('end_char', span.get('end_index')))
        if start is not None and end is not None:
            pred_spans.append((int(start), int(end)))

    return merge_adjacent(pred_spans)


def build_predictor(
    method,
    lettuce_model=DEFAULT_LETTUCE_MODEL,
    device=None,
    lookback_classifier_dir=DEFAULT_LOOKBACK_CLASSIFIER_DIR,
    lookback_model=None,
    lookback_sliding_window=None,
    lookback_threshold=None,
):
    if method == 'tool_overlap':
        return tool_overlap_predict
    if method == 'lettucedetect':
        predictor = LettuceDetectPredictor(
            model_name=lettuce_model,
            device=device,
        )
        return predictor.predict
    if method == 'lookback_lens':
        predictor = LookBackLensPredictorWrapper(
            classifier_dir=lookback_classifier_dir,
            lookback_model=lookback_model,
            device=device,
            sliding_window=lookback_sliding_window,
            threshold=lookback_threshold,
        )
        return predictor.predict
    raise NotImplementedError(f'Unknown method: {method}')


def evaluate(
    dataset_path,
    method='tool_overlap',
    lettuce_model=DEFAULT_LETTUCE_MODEL,
    device=None,
    lookback_classifier_dir=DEFAULT_LOOKBACK_CLASSIFIER_DIR,
    lookback_model=None,
    lookback_sliding_window=None,
    lookback_threshold=None,
):
    items = load_eval_items(dataset_path)
    print(
        f'Building predictor: method={method}, lettuce_model={lettuce_model}, '
        f'lookback_classifier={lookback_classifier_dir}, device={device}',
        flush=True,
    )
    predict = build_predictor(
        method,
        lettuce_model=lettuce_model,
        device=device,
        lookback_classifier_dir=lookback_classifier_dir,
        lookback_model=lookback_model,
        lookback_sliding_window=lookback_sliding_window,
        lookback_threshold=lookback_threshold,
    )
    print('Predictor is ready. Starting evaluation...', flush=True)
    tp = 0
    fp = 0
    fn = 0
    for ex in tqdm(items, desc=f'Evaluating {method}', unit='example'):
        gold = ex.get('hallucination_labels', [])
        gold_spans = [(g['start'], g['end']) for g in gold]
        pred_spans = predict(ex)

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
    return {'method': method, 'tp': tp, 'fp': fp, 'fn': fn, 'precision': prec, 'recall': rec, 'f1': f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to a RAGTruth-style JSONL file or a directory with generated JSONL files')
    parser.add_argument(
        '--method',
        default='tool_overlap',
        choices=['tool_overlap', 'lettucedetect', 'lookback_lens'],
        help='Baseline method',
    )
    parser.add_argument(
        '--lettuce_model',
        default=DEFAULT_LETTUCE_MODEL,
        help='Hugging Face model id or local path for LettuceDetect',
    )
    parser.add_argument(
        '--lookback_classifier',
        default=DEFAULT_LOOKBACK_CLASSIFIER_DIR,
        help='Directory with classifier.pkl and metadata.json for LookBack Lens',
    )
    parser.add_argument(
        '--lookback_model',
        default=None,
        help='Optional override for the LookBack Lens causal LM backbone',
    )
    parser.add_argument(
        '--lookback_sliding_window',
        type=int,
        default=None,
        help='Optional override for the LookBack Lens sliding window size',
    )
    parser.add_argument(
        '--lookback_threshold',
        type=float,
        default=None,
        help='Optional override for the LookBack Lens hallucination probability threshold',
    )
    parser.add_argument('--device', default=None, help='Device for neural baselines, e.g. cpu, cuda, cuda:0')
    args = parser.parse_args()
    evaluate(
        args.dataset,
        method=args.method,
        lettuce_model=args.lettuce_model,
        device=args.device,
        lookback_classifier_dir=args.lookback_classifier,
        lookback_model=args.lookback_model,
        lookback_sliding_window=args.lookback_sliding_window,
        lookback_threshold=args.lookback_threshold,
    )


if __name__ == '__main__':
    main()
