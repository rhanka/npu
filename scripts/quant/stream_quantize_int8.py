"""
Quantizer INT8 weight-only STREAMING — voie autonome bas-mémoire (sans Quark).

Quark OOM à ~19.4GB (4 copies du graphe). Ici on lit chaque poids un par un
depuis le fichier external_data (offset/length), on quantifie en INT8 par-canal
symétrique, et on insère un DequantizeLinear. Pic mémoire ≈ poids INT8 accumulés
(~2.3GB) + transitoire (~300MB) — très loin des 20GB.

Format : QDQ weight-only. Chaque poids FP16 [out, in] devient :
  W_int8 (int8, external) + W_scale (fp16, [out]) → DequantizeLinear(axis=0) → W (fp16)
Le MatMul/Gemm consommateur est inchangé (consomme toujours W).

Usage : python stream_quantize_int8.py [--src ...] [--dst ...]
"""
import argparse, os, struct
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/antoinefa/kog/onnx/out_fp16/laneformer.onnx")
    ap.add_argument("--dst", default="/home/antoinefa/kog/quant/out_int8_stream/laneformer_int8.onnx")
    ap.add_argument("--min-numel", type=int, default=4096,
                    help="ne quantifie que les poids >= ce nombre d'éléments (skip petits)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    data_path = args.src + ".data"

    print(f"[stream] chargement du graphe (sans data)…")
    model = onnx.load(args.src, load_external_data=False)
    g = model.graph

    # Index des initializers par nom
    init_by_name = {t.name: t for t in g.initializer}

    # Sélection : poids FP16 2D ET 3D (les 3D sont les shards DTP [8, out, in])
    targets = []
    for t in g.initializer:
        if t.data_type == TensorProto.FLOAT16 and len(t.dims) >= 2:
            numel = int(np.prod(t.dims))
            if numel >= args.min_numel:
                targets.append(t.name)
    print(f"[stream] {len(targets)} poids FP16 (2D+3D) à quantifier (sur {len(g.initializer)} initializers)")

    new_inits = []        # int8 + scale initializers
    new_nodes = []        # DequantizeLinear nodes
    quantized = set()

    with open(data_path, "rb") as f:
        for name in targets:
            t = init_by_name[name]
            ed = {e.key: e.value for e in t.external_data}
            offset, length = int(ed["offset"]), int(ed["length"])
            dims = [int(d) for d in t.dims]

            f.seek(offset)
            raw = f.read(length)
            w = np.frombuffer(raw, dtype=np.float16).reshape(dims).astype(np.float32)

            # INT8 symétrique PER-TENSOR (scale scalaire) — robuste pour les
            # shards DTP 3D [8, out, in] où un axe unique ne suffit pas.
            amax = float(np.abs(w).max())
            scale = max(amax / 127.0, 1e-8)
            q = np.round(w / scale).clip(-127, 127).astype(np.int8)

            w_int8 = numpy_helper.from_array(q, name=name + "_int8")
            w_scale = numpy_helper.from_array(
                np.array(scale, dtype=np.float16), name=name + "_scale")
            new_inits.append(w_int8)
            new_inits.append(w_scale)
            # DequantizeLinear scalaire (pas d'axis) → produit le tenseur `name`
            dq = helper.make_node(
                "DequantizeLinear",
                inputs=[name + "_int8", name + "_scale"],
                outputs=[name],
                name=name + "_dq",
            )
            new_nodes.append(dq)
            quantized.add(name)
            del w, q, raw

    print(f"[stream] {len(quantized)} poids quantifiés. Réécriture du graphe…")
    # DequantizeLinear avec scale/sortie FP16 nécessite opset >= 19 (on met 21).
    for op in model.opset_import:
        if op.domain in ("", "ai.onnx") and op.version < 21:
            op.version = 21
    # Retirer les initializers FP16 quantifiés, ajouter int8+scale
    keep = [t for t in g.initializer if t.name not in quantized]
    del g.initializer[:]
    g.initializer.extend(keep)
    g.initializer.extend(new_inits)
    # Les DequantizeLinear doivent précéder leurs consommateurs : on les met en tête
    old_nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes)
    g.node.extend(old_nodes)

    print(f"[stream] matérialisation des initializers gardés (petits, non-quantifiés)…")
    # Les tenseurs gardés (norms, biais) référencent encore l'ancien .data.
    # On charge leurs octets depuis le dossier source avant de tout réécrire.
    onnx.load_external_data_for_model(model, os.path.dirname(args.src))

    print(f"[stream] sauvegarde avec external data…")
    onnx.save(model, args.dst, save_as_external_data=True,
              all_tensors_to_one_file=True,
              location=os.path.basename(args.dst) + ".data")

    d = os.path.dirname(args.dst)
    mb = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)
             if os.path.isfile(os.path.join(d, f))) // 1024 // 1024
    print(f"[stream] ✅ INT8 écrit — {mb} MB (vs ~4400 MB FP16)")


if __name__ == "__main__":
    main()
