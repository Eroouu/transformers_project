# Dataset format (RAGTruth-style)

Each record is a JSON object with the following fields:

- `query`: user query (string)
- `context`: tool output used as context (string)
- `output`: model final response (string)
- `hallucination_labels`: list of span annotations we inject. Each annotation is an object with:
  - `start`: start character offset (inclusive) in `output`
  - `end`: end character offset (exclusive) in `output`
  - `label`: usually `hallucination`
  - `type`: one of `contradiction`, `overgeneration`, `missing_tool`
  - `text`: the substring that was annotated
- `source_output`: original uncorrupted assistant answer
- `corruption_type`: `clean`, `contradiction`, `overgeneration`, or `missing_tool`
- `corruption_strategy`: short name of the generation strategy used

The provided `data/generate_hallucinations.py` script writes four JSONL files:
- `clean.jsonl`
- `contradiction.jsonl`
- `overgeneration.jsonl`
- `missing_tool.jsonl`

Generation strategies:

- `clean` keeps the original assistant answer and uses an empty label list.
- `contradiction` parses JSON-like tool outputs, extracts scalar facts, and changes a value that appears in the assistant answer when possible. This produces tighter labels than appending generic text.
- `overgeneration` appends a plausible but unsupported domain-specific claim.
- `missing_tool` appends an action claim, such as booking, buying, transferring, or filing, that would require a tool call not present in the trace.
