# Hallucination Detection in Tool Calling

This repository contains code to create span-level hallucination datasets for dialogues involving tool calls, evaluate baseline detectors, and provide a starter span-classification training pipeline.

Overview

- Generate clean and corrupted datasets from a base ToolACE JSONL file or Hugging Face ToolACE:
  - Clean original examples with no hallucination labels
  - Contradiction between tool output and model response
  - Overgeneration (adds information not present in tool output)
  - Missing tool (response recommends actions requiring unavailable tools)
- Evaluate baseline detectors (heuristic baseline, LettuceDetect, and LookBack Lens)
- Train a span-level token-classification model to identify hallucinated spans

Quickstart

1. Create an environment and install dependencies:

Linux / macOS / WSL:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Windows (Command Prompt):

```cmd
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell prevents running scripts, enable temporary execution with:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
```

1. Generate corrupted datasets from the sample ToolACE JSONL:

```bash
python data/generate_hallucinations.py --input data/sample_toolace.jsonl --output_dir outputs
```

Or load the full ToolACE dataset directly from Hugging Face:

```bash
python data/generate_hallucinations.py --hf Team-ACE/ToolACE --hf_split train --output_dir outputs/toolace
```

For a quick smoke test on a few ToolACE examples:

```bash
python data/generate_hallucinations.py --hf Team-ACE/ToolACE --hf_split train --limit 10 --output_dir outputs/toolace_smoke
```

The generator writes four JSONL files:

- `clean.jsonl`: original model responses with empty `hallucination_labels`
- `contradiction.jsonl`: field-aware edits that change a tool-supported value when possible, such as a number, percentage, status, date, or weather value
- `overgeneration.jsonl`: domain-aware unsupported facts appended to the original answer
- `missing_tool.jsonl`: action claims that would require an unavailable or uncalled tool

Each corrupted record keeps `source_output`, `corruption_type`, and `corruption_strategy` metadata so examples can be audited later.

Validate generated datasets before training or evaluation:

```bash
python data/validate_corrupted_datasets.py outputs/toolace
```

Split the generated ToolACE dataset into aligned train/test folders:

```bash
python data/split_corrupted_datasets.py --input_dir outputs/toolace --train_dir outputs/toolace_train --test_dir outputs/toolace_test --test_ratio 0.2 --seed 42
```

1. Run the heuristic baseline evaluation:

```bash
python src/eval_baselines.py --dataset outputs/toolace/contradiction.jsonl --method tool_overlap
```

Run the LettuceDetect baseline:

```bash
python src/eval_baselines.py --dataset outputs/toolace/contradiction.jsonl --method lettucedetect --device cuda
```

By default this uses `KRLabsOrg/lettucedect-base-modernbert-en-v1`. To use another
checkpoint or force a specific device:

```bash
python src/eval_baselines.py --dataset outputs/toolace/contradiction.jsonl --method lettucedetect --lettuce_model KRLabsOrg/lettucedect-large-modernbert-en-v1 --device cuda
```

### LookBack Lens: training and evaluation

LookBack Lens is the second required baseline from the assignment. It extracts
**lookback ratios** from a causal LM's attention maps (how much the model attends to
tool output vs. the answer) and trains a lightweight **logistic-regression classifier**
to predict hallucinated spans.

**Important:** LookBack Lens is not zero-shot. You must train the classifier first,
then run evaluation with `--method lookback_lens`.

Default backbone LM: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (fits on an 8 GB GPU).
Training does **not** use epochs: the script makes one pass over the dataset to extract
attention features, then fits the classifier once.

#### Quick smoke test (recommended first)

```bash
python src/train_lookback_lens.py \
  --dataset outputs/toolace_smoke \
  --output_dir models/lookback_lens_smoke \
  --device cuda \
  --limit 8

python src/eval_baselines.py \
  --dataset outputs/toolace_smoke/contradiction.jsonl \
  --method lookback_lens \
  --lookback_classifier models/lookback_lens_smoke \
  --device cuda
```

#### Full training on the train split

Make sure you already generated and split the dataset (steps 2 above). Then:

```bash
python src/train_lookback_lens.py  --dataset outputs/toolace_train  --output_dir models/lookback_lens  --device cuda
```

This saves two files in `models/lookback_lens/`:

- `classifier.pkl` — trained logistic-regression classifier
- `metadata.json` — backbone model name, sliding window size, threshold

The causal LM itself is **not** saved; it is reloaded from Hugging Face during
evaluation using the name stored in `metadata.json`.

Useful training flags:


| Flag               | Default                              | Description                               |
| ------------------ | ------------------------------------ | ----------------------------------------- |
| `--model`          | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Causal LM for attention extraction        |
| `--sliding_window` | `8`                                  | Token window size for span classification |
| `--threshold`      | `0.5`                                | Hallucination probability threshold       |
| `--limit`          | none                                 | Cap examples for a quick debug run        |
| `--device`         | `auto`                               | `cuda`, `cpu`, or `auto`                  |


#### Evaluation on the test split

Evaluate on one corruption type:

```bash
python src/eval_baselines.py \
  --dataset outputs/toolace_test/contradiction.jsonl \
  --method lookback_lens \
  --lookback_classifier models/lookback_lens \
  --device cuda
```

Repeat for `overgeneration.jsonl`, `missing_tool.jsonl`, and `clean.jsonl`.

Evaluate on all four files at once (directory loads `clean`, `contradiction`,
`overgeneration`, and `missing_tool`):

```bash
python src/eval_baselines.py --dataset outputs/toolace_test --method lookback_lens --lookback_classifier models/lookback_lens --device cuda
```

Optional evaluation flags:


| Flag                        | Default                | Description                     |
| --------------------------- | ---------------------- | ------------------------------- |
| `--lookback_classifier`     | `models/lookback_lens` | Directory with `classifier.pkl` |
| `--lookback_model`          | from metadata          | Override backbone LM            |
| `--lookback_sliding_window` | from metadata          | Override window size            |
| `--lookback_threshold`      | from metadata          | Override decision threshold     |


Example output:

```text
Method=lookback_lens  TP=... FP=... FN=... P=... R=... F1=...
```

1. (Optional) Train a small span-classification model:

```bash
python src/train_span_model.py --dataset outputs/toolace/contradiction.jsonl --output_dir models/span_model
```

Fine-tune a LettuceDetect-compatible model on the generated datasets:

```bash
python src/train_lettucedetect.py --dataset outputs/toolace --output_dir models/lettucedetect_toolace --device cuda --fp16 --batch_size 1 --gradient_accumulation_steps 8 --gradient_checkpointing
```

Evaluate the fine-tuned checkpoint with the LettuceDetect baseline wrapper:

```bash
python src/eval_baselines.py --dataset outputs/toolace/contradiction.jsonl --method lettucedetect --lettuce_model models/lettucedetect_toolace --device cuda
```

Train on the aligned ToolACE train split used in the latest experiment:

```bash
python src/train_lettucedetect.py --dataset outputs/toolace_train --output_dir models/lettucedetect_toolace_train_eval_fast --device cuda --fp16 --batch_size 1 --eval_batch_size 2 --gradient_accumulation_steps 4
```

Evaluate the checkpoint on the full held-out test directory:

```bash
python src/eval_baselines.py --dataset outputs/toolace_test --method lettucedetect --lettuce_model models/lettucedetect_toolace_train_eval_fast --device cuda
```

Latest local result on `outputs/toolace_test`:

```text
Loaded 1092 examples from outputs/toolace_test
Method=lettucedetect  TP=796 FP=32 FN=23 P=0.9614 R=0.9719 F1=0.9666
Method=lookback_lens  TP=799 FP=1110 FN=20 P=0.4185 R=0.9756 F1=0.5858
```

The evaluation script also accepts a single JSONL file, for example:

```bash
python src/eval_baselines.py --dataset outputs/toolace_test/contradiction.jsonl --method lettucedetect --lettuce_model models/lettucedetect_toolace_train_eval_fast --device cuda
```

Repository structure

- `data/` — sample ToolACE JSONL and generation script
- `src/` — evaluation and training scripts (`lookback_lens.py`, `train_lookback_lens.py`)
- `models/` — saved classifiers and fine-tuned checkpoints (gitignored)
- `experiments/` — run scripts
- `docs/` — dataset format and design notes

Next steps

- Run experiments and publish dataset/model on Hugging Face

License: MIT