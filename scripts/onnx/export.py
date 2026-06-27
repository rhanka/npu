"""
Export Laneformer 2B → ONNX.

Spike A concluant : torch.onnx.export (dynamo) fonctionne dès lors que le RoPE
complexe (torch.polar / view_as_complex) est remplacé par une rotation réelle
sin/cos équivalente (voir rope_patch.py). SWA et DTP ne posent aucun problème :
- SWA : masques standard via create_sliding_window_causal_mask (traçables)
- DTP : absent du forward mono-device

Usage:
    python export.py [--dtype fp32|fp16] [--out OUT_DIR]

Le modèle exporté produit des poids externes (.onnx.data) car > 2 Go.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rope_patch import patch_model_for_onnx

MODEL_ID = "kogai/laneformer-2b-it"
REVISION = "b4f40adc413c2c5268ab89cf666ade37148d8d4b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out"))
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    os.makedirs(args.out, exist_ok=True)

    print(f"[export] chargement {MODEL_ID} ({args.dtype})…")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True,
        dtype=dtype, device_map="cpu",
    )
    model.eval()
    patch_model_for_onnx(model)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    inputs = tok("Hello world, this is a test", return_tensors="pt")

    out_path = os.path.join(args.out, "laneformer.onnx")
    print(f"[export] torch.onnx.export → {out_path}")
    torch.onnx.export(
        model,
        (inputs["input_ids"],),
        out_path,
        opset_version=args.opset,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "logits": {0: "batch", 1: "seq"}},
        dynamo=True,
    )
    size = os.path.getsize(out_path) // 1024 // 1024
    print(f"[export] ✅ {size} MB (graphe) + poids externes .onnx.data")


if __name__ == "__main__":
    main()
