# Quantification INT8 streaming — RÉSULTAT (2026-06-27)

**Voie autonome bas-mémoire** : `stream_quantize_int8.py`. Contourne l'OOM d'AMD
Quark (pic 19.4GB, ~4 copies du graphe) en lisant chaque poids un par un depuis
le fichier external_data (offset/length) et en le quantifiant en INT8.

## Méthode
- Graphe chargé sans data (~tens de MB) ; poids lus en streaming via offset/length
- INT8 symétrique **per-tensor** (scale scalaire) — robuste pour les shards DTP 3D `[8, out, in]`
- Insertion d'un `DequantizeLinear` (FP16, opset 21) par poids → MatMul/Gemm inchangés
- 107 poids quantifiés (2D + 3D) sur 165 initializers

## Résultats
| Métrique | Valeur |
|---|---|
| Taille INT8 | **2216 MB** (vs 4400 MB FP16, ~50%) |
| Pic RSS quantif | **4.5 GB** (vs 19.4 GB OOM de Quark) |
| Validation top-1 | ✅ "Paris" == "Paris" |
| top-5 overlap | 5/5 |

## Limites / suite
- **Weight-only** : active uniquement la quantif des poids, pas des activations.
  Pour le NPU XINT8 complet (activations INT8 + calibration), il faut AMD Quark
  (bloqué par OOM mémoire — cf. quark-oom-blocker, nécessite +swap).
- Per-tensor (pas per-canal) à cause des shards DTP 3D ; per-canal possible en
  dépliant les shards mais complexifie le DequantizeLinear.
- Prochaine étape : runtime GPU (onnxruntime-rocm) ou NPU (VitisAI EP) pour
  bénéficier de l'accélération INT8.
