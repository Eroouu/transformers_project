# Hallucination Detection in Tool Calling

This repository contains code to create span-level hallucination datasets for dialogues involving tool calls, evaluate baseline detectors, and provide a starter span-classification training pipeline.

Overview
- Generate clean and corrupted datasets from a base ToolACE JSONL file or Hugging Face ToolACE:
  - Clean original examples with no hallucination labels
  - Contradiction between tool output and model response
  - Overgeneration (adds information not present in tool output)
  - Missing tool (response recommends actions requiring unavailable tools)
- Evaluate baseline detectors (heuristic baseline provided; placeholders for LettuceDetect and LookBackLens)
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

2. Generate corrupted datasets from the sample ToolACE JSONL:

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

3. Run the heuristic baseline evaluation:

```bash
python src/eval_baselines.py --dataset outputs/toolace/contradiction.jsonl --method tool_overlap
```

4. (Optional) Train a small span-classification model:

```bash
python src/train_span_model.py --dataset outputs/toolace/contradiction.jsonl --output_dir models/span_model
```

Repository structure
- `data/` — sample ToolACE JSONL and generation script
- `src/` — evaluation and training scripts
- `experiments/` — run scripts
- `docs/` — dataset format and design notes

Next steps
- Implement wrappers for LettuceDetect and LookBackLens
- Run experiments and publish dataset/model on Hugging Face

License: MIT
