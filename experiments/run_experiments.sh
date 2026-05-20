#!/usr/bin/env bash
set -euo pipefail

python data/generate_hallucinations.py --input data/sample_toolace.jsonl --output_dir outputs
python src/eval_baselines.py --dataset outputs/contradiction.jsonl --method tool_overlap
python src/train_span_model.py --dataset outputs/contradiction.jsonl --output_dir models/span_model
