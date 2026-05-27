# Hallucination Detection in Tool Calling

This repository builds span-level hallucination datasets for tool-calling dialogues, evaluates baseline detectors, and trains an improved span detector for the final assignment.

The task uses ToolACE-style examples with:

- user query
- available tools
- tool call and tool output
- final assistant answer

The final dataset follows a RAGTruth-style schema:

- `query`: user query
- `context`: tool output
- `output`: assistant final answer
- `hallucination_labels`: character-level hallucination spans
- `corruption_type`: `clean`, `contradiction`, `overgeneration`, or `missing_tool`

## Current Artifacts

The main prepared dataset is:

- `final_dataset/`
- `final_dataset_train/`
- `final_dataset_test/`

Each dataset folder contains:

- `clean.jsonl`
- `contradiction.jsonl`
- `overgeneration.jsonl`
- `missing_tool.jsonl`

Validated final counts:

| split | records per file | total records |
| --- | ---: | ---: |
| `final_dataset` | 2431 | 9724 |
| `final_dataset_train` | 1945 | 7780 |
| `final_dataset_test` | 486 | 1944 |

## Setup

Linux / macOS / Colab:

```bash
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `torchvision` causes a ModernBERT import error in Colab, uninstall it because this project is text-only:

```bash
pip uninstall -y torchvision
```

Use forward slashes in Colab/Linux paths, for example `models/lookback_lens_final`, not `models\lookback_lens_final`.

## Dataset Generation

Build clean ToolACE records:

```bash
python data/build_toolace_clean.py --hf Team-ACE/ToolACE --hf_split train --output_dir outputs/toolace
```

Generate LLM-patched corruptions:

```bash
python data/generate_llm_corruption.py --input outputs/toolace/clean.jsonl --output outputs/toolace/contradiction.jsonl --corruption_type contradiction --model gpt-4o-mini --temperature 0.2 --overwrite
python data/generate_llm_corruption.py --input outputs/toolace/clean.jsonl --output outputs/toolace/overgeneration.jsonl --corruption_type overgeneration --model gpt-4o-mini --temperature 0.3 --overwrite
python data/generate_llm_corruption.py --input outputs/toolace/clean.jsonl --output outputs/toolace/missing_tool.jsonl --corruption_type missing_tool --model gpt-4o-mini --temperature 0.3 --overwrite
```

Validate a dataset folder:

```bash
python data/validate_corrupted_datasets.py final_dataset
python data/validate_corrupted_datasets.py final_dataset_train
python data/validate_corrupted_datasets.py final_dataset_test
```

The older `data/generate_hallucinations.py` script is kept for rule-based, hybrid, and legacy synthetic generation.

## Baselines

### Tool Overlap

Simple heuristic baseline:

```bash
python src/eval_baselines.py --dataset final_dataset_test --method tool_overlap
```

### LettuceDetect

```bash
python src/eval_baselines.py --dataset final_dataset_test --method lettucedetect --device cuda
```

CPU fallback:

```bash
python src/eval_baselines.py --dataset final_dataset_test --method lettucedetect --device cpu
```

### LookBack Lens

LookBack Lens is not zero-shot in this repo. Train the attention-feature classifier first:

```bash
python src/train_lookback_lens.py --dataset final_dataset_train --output_dir models/lookback_lens_final --device cuda
```

Then evaluate:

```bash
python src/eval_baselines.py --dataset final_dataset_test --method lookback_lens --lookback_classifier models/lookback_lens_final --device cuda
```

## Improved Span Model

The improved model is implemented in:

- `src/finetuned_span_model.py`

It fine-tunes a transformer token classifier on ToolACE-style hallucination spans. The model receives the user query, tool output, and answer, but computes loss only over answer tokens. It uses class weighting, focal loss, validation threshold tuning, and span post-processing.

Train:

```bash
python src/finetuned_span_model.py train \
  --dataset_dir final_dataset_train \
  --output_dir models/finetuned_span_model \
  --device cuda \
  --fp16 \
  --gradient_checkpointing
```

Evaluate:

```bash
python src/finetuned_span_model.py evaluate \
  --dataset final_dataset_test \
  --model_dir models/finetuned_span_model \
  --device cuda \
  --metrics_out outputs/finetuned_span_model_metrics.json
```

Evaluate one corruption type:

```bash
python src/finetuned_span_model.py evaluate --dataset final_dataset_test/contradiction.jsonl --model_dir models/finetuned_span_model --device cuda
python src/finetuned_span_model.py evaluate --dataset final_dataset_test/overgeneration.jsonl --model_dir models/finetuned_span_model --device cuda
python src/finetuned_span_model.py evaluate --dataset final_dataset_test/missing_tool.jsonl --model_dir models/finetuned_span_model --device cuda
python src/finetuned_span_model.py evaluate --dataset final_dataset_test/clean.jsonl --model_dir models/finetuned_span_model --device cuda
```

## Results

Evaluation is span-level. A predicted span is counted as a true positive if it overlaps a gold hallucination span. There is no span-level true-negative count because the number of non-hallucinated spans is not well-defined.

### Overall

| model | tp | fp | fn | precision | recall | f1 | n_examples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tool_overlap | 2099 | 13582 | 18 | 0.133900 | 0.991500 | 0.235900 | 1944 |
| lettucedetect | 1116 | 3457 | 474 | 0.244000 | 0.701900 | 0.362200 | 1944 |
| lookback_lens | 1425 | 2623 | 38 | 0.352000 | 0.974000 | 0.517100 | 1944 |
| finetuned_span_model | 1376 | 113 | 90 | 0.924100 | 0.938600 | 0.931300 | 1944 |

### Contradiction

| model | tp | fp | fn | precision | recall | f1 | n_examples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ensemble_weighted_vote | 451 | 583 | 48 | 0.436170 | 0.903808 | 0.588389 | 486 |
| lettucedetect | 376 | 823 | 121 | 0.313600 | 0.756500 | 0.443400 | 486 |
| lookback_lens | 452 | 553 | 36 | 0.449800 | 0.926200 | 0.605500 | 486 |
| finetuned_span_model | 403 | 42 | 86 | 0.905600 | 0.824100 | 0.863000 | 486 |

### Overgeneration

| model | tp | fp | fn | precision | recall | f1 | n_examples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ensemble_weighted_vote | 586 | 636 | 35 | 0.479542 | 0.943639 | 0.635920 | 486 |
| lettucedetect | 451 | 837 | 72 | 0.350200 | 0.862300 | 0.498100 | 486 |
| lookback_lens | 487 | 615 | 1 | 0.441900 | 0.998000 | 0.612600 | 486 |
| finetuned_span_model | 489 | 10 | 2 | 0.980000 | 0.995900 | 0.987900 | 486 |

### Missing Tool

| model | tp | fp | fn | precision | recall | f1 | n_examples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ensemble_weighted_vote | 435 | 659 | 116 | 0.397623 | 0.789474 | 0.528875 | 486 |
| lettucedetect | 289 | 900 | 281 | 0.243100 | 0.507000 | 0.328600 | 486 |
| lookback_lens | 486 | 621 | 1 | 0.439000 | 0.997900 | 0.609800 | 486 |
| finetuned_span_model | 484 | 7 | 2 | 0.985700 | 0.995900 | 0.990800 | 486 |

### Clean

The clean split contains no gold hallucination spans. The main quantity is therefore false positives.

| model | tp | fp | fn | n_examples |
| --- | ---: | ---: | ---: | ---: |
| finetuned_span_model | 0 | 54 | 0 | 486 |

## Repository Structure

- `data/`: dataset construction, corruption generation, validation, splitting, and merging
- `src/eval_baselines.py`: baseline evaluation harness
- `src/train_lookback_lens.py` and `src/lookback_lens.py`: LookBack Lens implementation
- `src/train_lettucedetect.py`: LettuceDetect-compatible fine-tuning script
- `src/finetuned_span_model.py`: improved span detector used for the final run
- `docs/`: dataset format notes
- `models/`: local model checkpoints, gitignored

## Next Steps

- Publish the final dataset on Hugging Face.
- Publish the fine-tuned span model on Hugging Face.
- Add contradiction-focused training examples to improve contradiction recall.
- Try a larger ModernBERT/LettuceDetect base model or a multi-seed ensemble for leaderboard optimization.

