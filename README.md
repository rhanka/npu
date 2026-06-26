# rhanka/npu

**NPU-optimized LLM inference on AMD Ryzen AI (XDNA2, up to 32 GB unified RAM)**

Recipes, benchmarks, and quantization pipelines for adapting open models to the Ryzen AI NPU.
Two primary use-cases: **GraphRAG / ontological indexing** (Graphify) and **coding assistance**.
Quantized models published on HuggingFace when benchmarks pass quality gates.

## Hardware target

| Component | Spec |
|---|---|
| NPU | Ryzen AI XDNA2 — ~50 TOPS INT8, `/dev/accel0` |
| GPU | Radeon 8060S gfx1151 — ROCm 7.2 (RDNA 3.5, INT8/AWQ) |
| Unified RAM | 57 GB LPDDR5X (up to 32 GB for inference) |
| OS | Ubuntu 26.04 + Ryzen AI Software 1.7.1 |

> gfx1151 = RDNA 3.5 — no hardware FP8. Use INT8/AWQ.

## Use-cases

### GraphRAG / ontological indexing (Graphify)
Structured output: entity extraction, relationship triples, ontology grounding.
Low latency over ≤ 4096 tokens.

### Coding assistance
Code generation + completion. Lightweight benchmark profile (HumanEval, EvalPlus, MBPP subset).

## Repo structure

```
models/<name>/
  recipe.yaml        # quant recipe (method, calibration, targets)
  MODELCARD.md       # HF-style model card
  benchmarks/        # per-model results (JSON)
benchmarks/
  suites/coding/     # HumanEval, EvalPlus, MBPP configs
  suites/graphrag/   # ontological extraction benchmark
  results/           # aggregated versioned results
  leaderboard.md
spec/                # design specs (STUDY → VOL → EVOL)
scripts/             # export, quant, eval, HF publish
.github/ISSUE_TEMPLATE/
```

## Benchmark standards

Results follow [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) format.

### Coding suite (local, lightweight)
| Benchmark | Samples | Metric |
|---|---|---|
| HumanEval | 164 | pass@1 |
| EvalPlus (HumanEval+) | 164 | pass@1 |
| MBPP subset | 100 | pass@1 |

### GraphRAG suite
| Benchmark | Metric |
|---|---|
| Graphify-synth-v1 (FR/EN ontology extraction) | F1 triples |
| JSON-struct (structured output fidelity) | exact match |
| Long-ctx-4k (faithfulness ≤ 4096 tokens) | ROUGE-L + LLM-judge |

### Quality gates before HF publish
- ΔPPL ≤ 1 % vs BF16 baseline (including SWA boundary > 512 tokens)
- Coding: pass@1 degradation ≤ 2 pp
- GraphRAG: F1 degradation ≤ 3 pp

## Models

| Model | Backend | Status | tok/s | HF |
|---|---|---|---|---|
| laneformer-2b-it BF16 | GPU ROCm | ✅ baseline | 16.7 | [kogai/laneformer-2b-it](https://huggingface.co/kogai/laneformer-2b-it) |
| laneformer-2b-it INT8 AWQ | GPU ROCm | 🔧 in progress | ~35 | — |
| laneformer-2b-it XINT8 | NPU XDNA2 | 📋 planned | perf/W | — |

## License

Apache 2.0 — model weights follow their upstream licenses.
