"""
Inférence Laneformer INT8 sur le NPU Ryzen AI (XDNA2) via VitisAI EP.

Prérequis (cf. scripts/setup-npu.sh) :
  - XRT + amdxdna installés, /dev/accel0 accessible
  - Ryzen AI Software 1.7.1 (fournit onnxruntime + VitisAIExecutionProvider + voe)
  - venv : /home/antoinefa/kog/quant/.venv-npu (créé par install_ryzen_ai.sh)

Le VitisAI EP compile le graphe ONNX en exécutable micro-codé pour le NPU au
démarrage de la session (mise en cache). Modèle d'entrée : INT8 ou BF16.

Usage : <.venv-npu>/bin/python scripts/npu_infer.py
"""
import os, sys, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np

INT8_MODEL = "/home/antoinefa/kog/quant/out_int8_stream/laneformer_int8.onnx"
CACHE_DIR = "/home/antoinefa/kog/quant/vitisai_cache"


def main():
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"[npu] providers disponibles : {providers}")
    if "VitisAIExecutionProvider" not in providers:
        print("[npu] ❌ VitisAIExecutionProvider absent — installer Ryzen AI Software")
        print("[npu]    (cf. scripts/setup-npu.sh). Le runtime Linux est gaté + bug #341.")
        sys.exit(1)

    os.makedirs(CACHE_DIR, exist_ok=True)
    so = ort.SessionOptions()
    # VitisAI EP : compile + cache le graphe pour le NPU
    provider_options = [{
        "config_file": os.environ.get("VAIP_CONFIG", ""),
        "cache_dir": CACHE_DIR,
        "cache_key": "laneformer_int8",
    }]
    print("[npu] création session VitisAI EP (compilation NPU au 1er run)…")
    t0 = time.time()
    sess = ort.InferenceSession(
        INT8_MODEL, sess_options=so,
        providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        provider_options=provider_options,
    )
    print(f"[npu] session prête en {time.time()-t0:.1f}s")

    # Inférence de test (prompt → next token)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("kogai/laneformer-2b-it", trust_remote_code=True)
        ids = tok("The capital of France is", return_tensors="pt")["input_ids"].numpy().astype(np.int64)
    except Exception:
        # fallback sans transformers : tokens arbitraires
        ids = np.array([[1, 450, 7483, 310, 3444, 338]], dtype=np.int64)
        tok = None

    t0 = time.time()
    logits = sess.run(["logits"], {"input_ids": ids})[0]
    dt = time.time() - t0
    nxt = int(logits[0, -1].argmax())
    txt = tok.decode([nxt]) if tok else str(nxt)
    print(f"[npu] ✅ inférence NPU OK — next token: {txt!r}  ({dt*1000:.0f}ms)")
    print(f"[npu] (attendu 'Paris' si offload NPU correct)")


if __name__ == "__main__":
    main()
