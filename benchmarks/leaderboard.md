# NPU Leaderboard

Results on Ryzen AI XDNA2 / gfx1151 (ROCm 7.2). Updated on each model publish.

| Model | Format | Size | tok/s | Status | Validation |
|---|---|---|---|---|---|
| laneformer-2b-it | BF16 (PyTorch GPU) | 4.6 GB | 16.7 | ✅ baseline | ref |
| laneformer-2b-it | ONNX FP16 | 4.4 GB | — | ✅ exported | top-1 100% vs PyTorch |
| laneformer-2b-it | ONNX INT8 (weight-only, streaming) | 2.2 GB | — | ✅ quantized | top-1 "Paris" + 5/5 top-5 vs FP16 |
| laneformer-2b-it | ONNX XINT8 (NPU, activations) | ~2.3 GB | — | 🔧 Quark OOM (needs +swap) | — |

_INT8 streaming quantizer (`scripts/quant/stream_quantize_int8.py`) bypasses AMD Quark's
OOM (19.4 GB peak) by streaming weights one at a time — peak 4.5 GB._
