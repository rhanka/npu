# kog-npu

Laneformer 2B — ONNX export + AMD Quark quantization pipeline targeting Ryzen AI NPU (XDNA2) and GPU INT8.

## Status

| Backend | Target | Status |
|---|---|---|
| GPU BF16 (baseline) | 16.7 tok/s | ✅ Running |
| GPU INT8/AWQ (ROCm) | ~33–40 tok/s | 🔧 In progress |
| NPU XDNA2 (VitisAI EP) | perf/watt goal | 📋 Planned |

## Architecture

- Model: `kogai/laneformer-2b-it` — Llama-variant, trust_remote_code, ~2.3B BF16
- Layers 0–9: SWA (Sliding Window Attention, window=512)
- Layers 10–14: Full-attention
- Hardware: ROG Z13 — Radeon 8060S gfx1151 (ROCm 7.2) + Ryzen AI NPU XDNA2

## Spec

See [`spec/SPEC_STUDY_LANEFORMER_ONNX.md`](spec/SPEC_STUDY_LANEFORMER_ONNX.md) for the full study including options, decisions, and peer consensus.

## Key decisions (panel consensus — Opus 4.8 xhigh)

- **D1**: GPU INT8/AWQ first (NPU = energy efficiency goal, not throughput)
- **D2**: spike `torch.onnx.export` eager (2h) → fallback manual forward decomposition; skip Optimum
- **D3**: SWA as inline static mask (NPU-compatible, no custom op)
- **D4**: 2 ONNX graphs — prefill + decode (stateful KV-cache)
- **D5**: ΔPPL ≤ 0.5% + SWA→full-attention boundary test (>512 tokens)

> Note: gfx1151 = RDNA 3.5, not RDNA4 — no hardware FP8. Use INT8 WMMA.
