# Spike A — Résultats export ONNX Laneformer 2B

**Date :** 2026-06-27
**Verdict :** ✅ **Option A (export direct) viable** — Option C (réécriture forward) non nécessaire.

## Démarche

1. Inspection du code `modeling_laneformer.py` (trust_remote_code)
2. Tentative `torch.onnx.export(dynamo=True)` → échec ciblé
3. Identification du blocage unique → patch → re-export → validation ONNX Runtime

## Le seul blocage : RoPE complexe

L'export passe `torch.export` ✅ et les décompositions ✅, et bloque à la
traduction ONNX sur :

```
aten.unsqueeze on %polar — No decompositions registered for the complex-valued input
```

Cause : le RoPE utilise l'arithmétique complexe (style Meta-Llama) :

```python
freqs_cis = torch.polar(torch.ones_like(angles), angles)   # complex64
xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
```

ONNX n'a pas de dtype complexe. **Ce n'est ni la SWA ni le DTP qui bloquent :**
- **SWA** : gérée par masques standard (`create_sliding_window_causal_mask`),
  parfaitement traçable. `swa_layers = [0..9]`, `sliding_window = 2048`.
- **DTP** : absent du forward mono-device, aucun `process_group` dans le trace.

## Le fix : RoPE réel équivalent (`rope_patch.py`)

La rotation complexe `(x0 + i·x1)·(cos + i·sin)` est strictement égale à :

```
o0 = x0·cos − x1·sin
o1 = x0·sin + x1·cos
```

On patche `_compute_freqs_cis` pour renvoyer les angles réels (même rang de
tenseur, réutilise `compute_rope_freqs` du modèle) et `apply_rotary_emb` pour
des ops réelles uniquement.

## Validation (gate D5)

| Comparaison | max abs diff | top-1 match |
|---|---|---|
| Patch RoPE réel vs complexe (PyTorch) | 1.0e-5 | 100 % |
| ONNX Runtime vs PyTorch référence | 1.5e-5 | 100 % |

Prompt « The capital of France is » → PyTorch et ONNX prédisent tous deux `Paris`.

## Artefacts

- `laneformer.onnx` : 3 MB (graphe)
- `laneformer.onnx.data` : 8.85 GB (poids FP32 externes ; FP16 ≈ 4.4 GB)

## Décisions confirmées

- **D2** : Option A retenue (export direct + patch RoPE). Skip Optimum (B) ET
  réécriture manuelle (C).
- **D3** : SWA via masque standard inline — déjà le cas dans le modèle, rien à faire.

## Reste à faire

1. Export **prefill + decode avec KV-cache** (`past_key_values`) — l'export
   actuel est full-context sans cache (D4 : 2 graphes stateful).
2. Quantification **AMD Quark INT8/AWQ** sur le ONNX FP16.
3. Validation **frontière SWA** : prompt > 2048 tokens (transition couches 9→10).
4. Compilation **VitisAI EP** pour le NPU (le vrai mur, cf. consensus P2).

---

## Mise à jour — KV-cache (D4) ✅ 2026-06-27

Export ONNX **avec KV-cache** réussi et validé end-to-end.

### Obstacles levés
1. `dynamic_axes` non supporté par dynamo → `dynamic_shapes` (torch.export.Dim).
2. `*past` varargs → 1 élément top-level (tuple de 30 dicts), pas 33.
3. `scaled_dot_product_attention is_causal SymBool` → **forcer `attn_implementation="eager"`**
   (matmul/softmax explicites, intégré au patch). Le modèle custom n'honore pas
   l'arg à l'init → forcé sur config + sous-modules.
4. seq spécialisé à 1 (torch.export squeeze les dims taille 1) → exemple
   non-dégénéré T_new=3, past_seq=2.

### Artefacts
- `kv_wrapper.py` : wrapper tuple-de-tenseurs ↔ DynamicCache (transformers 5.12.1)
- `export_kv.py` : graphe unique prefill (past_seq=0) + decode (past_seq>0)
- 30 tenseurs cache E/S (15 couches × k,v), axes batch/seq/past_seq dynamiques

### Validation
| Test | Résultat |
|---|---|
| Wrapper KV vs full-recompute (PyTorch) | max diff 1.43e-5 sur 8 steps |
| ONNX KV (ORT) vs PyTorch, prefill→decode 8 tokens | **100% token match** |

Sortie identique : « Paris, and the capital of Spain is ».

### Reste
- Quantification AMD Quark INT8/AWQ sur le ONNX KV
- Frontière SWA > 2048 tokens (transition couches 9→10)
- Runtime : ORT ROCm EP (GPU) puis VitisAI EP (NPU)
