#!/usr/bin/env bash
# Wrapper around lm-evaluation-harness for NPU/GPU quantized models.
# Usage: ./eval_harness.sh <model_path> <task> [--limit N]
set -euo pipefail
MODEL="${1:?model path required}"
TASK="${2:?task required (humaneval|mbpp|graphrag-synth)}"
LIMIT="${3:-}"
python -m lm_eval --model hf \
  --model_args "pretrained=$MODEL,trust_remote_code=True" \
  --tasks "$TASK" \
  --output_path benchmarks/results/ \
  ${LIMIT:+--limit $LIMIT}
