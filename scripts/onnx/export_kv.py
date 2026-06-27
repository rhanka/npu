"""
Export Laneformer 2B → ONNX AVEC KV-cache (D4).

Graphe unique gérant prefill (T_past=0) et decode (T_past>0) via axes dynamiques.
Entrées : input_ids, position_ids, cache_position, past_{k,v}_{0..14}
Sorties : logits, present_{k,v}_{0..14}

Usage: python export_kv.py [--dtype fp32|fp16] [--out OUT_DIR]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rope_patch import patch_model_for_onnx
from kv_wrapper import LaneformerKVWrapper, empty_past

MODEL_ID = "kogai/laneformer-2b-it"
REVISION = "b4f40adc413c2c5268ab89cf666ade37148d8d4b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out_kv"))
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    os.makedirs(args.out, exist_ok=True)

    # attn eager : masques explicites, pas de SDPA is_causal SymBool (export dynamique)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True, dtype=dtype,
        device_map="cpu", attn_implementation="eager")
    model.eval()
    patch_model_for_onnx(model)
    wrapper = LaneformerKVWrapper(model); wrapper.eval()

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    ids = tok("Hello world test", return_tensors="pt")["input_ids"]
    T = ids.shape[1]
    n_layers = model.config.num_hidden_layers

    # Exemple DECODE non-dégénéré (past_seq=2, T_new=3) : des dims de taille > 1
    # évitent que torch.export ne spécialise/squeeze le seq à 1. Au runtime,
    # seq et past_seq sont dynamiques (prefill = past_seq 0, decode = T_new 1).
    T_PAST = 2
    T_NEW = 3
    n_kv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    past = []
    for _ in range(model.config.num_hidden_layers):
        past.append(torch.zeros(1, n_kv, T_PAST, head_dim, dtype=dtype))
        past.append(torch.zeros(1, n_kv, T_PAST, head_dim, dtype=dtype))
    ids = ids[:, :T_NEW]
    T = T_NEW
    pos = torch.arange(T_PAST, T_PAST + T_NEW).unsqueeze(0)
    cpos = torch.arange(T_PAST, T_PAST + T_NEW)

    # Noms d'E/S
    past_names = []
    for i in range(n_layers):
        past_names += [f"past_k_{i}", f"past_v_{i}"]
    present_names = []
    for i in range(n_layers):
        present_names += [f"present_k_{i}", f"present_v_{i}"]
    input_names = ["input_ids", "position_ids", "cache_position"] + past_names
    output_names = ["logits"] + present_names

    # dynamic_shapes (format dynamo natif) : seq et past_seq dynamiques.
    from torch.export import Dim
    seq = Dim("seq", min=1, max=4096)
    past_seq = Dim("past_seq", min=0, max=4096)
    # *past varargs → un seul élément top-level (liste de 30 dicts)
    dynamic_shapes = (
        {1: seq},      # input_ids [B, seq]
        {1: seq},      # position_ids
        {0: seq},      # cache_position
        tuple({2: past_seq} for _ in past_names),  # *past (tuple, comme varargs)
    )

    out_path = os.path.join(args.out, "laneformer_kv.onnx")
    print(f"[export-kv] torch.onnx.export → {out_path}")
    torch.onnx.export(
        wrapper,
        (ids, pos, cpos, *past),
        out_path,
        opset_version=args.opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
    )
    size = os.path.getsize(out_path) // 1024 // 1024
    print(f"[export-kv] ✅ {size} MB (graphe) + poids externes")


if __name__ == "__main__":
    main()
