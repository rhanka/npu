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

---

## Mise à jour — FP16 + quantification INT8 (2026-06-27)

### FP16 KV export ✅
`export_kv.py --dtype fp16` → **4.4 GB** (moitié du FP32 8.7 GB).
ORT validé : sortie identique « Paris, and the capital of Spain is ».
Format déployable, base correcte pour AMD Quark et ORT ROCm EP.

### INT8 via ORT quantize_dynamic ❌ — voie morte ici
- Depuis FP32 (8.7 GB) : **OOM** (cap 32 GB, charge tout le modèle en RAM).
- Depuis FP16 (4.4 GB) : produit un graphe **invalide** + **aucune réduction**
  (4335 MB ≈ 4400 MB — quantize_dynamic exige du FP32 pour quantifier les poids).
- De plus, `quantize_dynamic` cible le **CPUExecutionProvider** (dequant runtime),
  pas l'accélération GPU/NPU visée.

### Conclusion : INT8 passe par AMD Quark, pas ORT
- `amd-quark` sur PyPI = placeholder (0.1.0), pas le vrai package.
- Le vrai Quark : canal AMD/ROCm, **incompatible Python 3.14** (besoin ≤ 3.12).
- C'est de toute façon la voie NPU correcte : Quark XINT8 → ONNX → VitisAI EP.

**Prochaine étape quantification** : environnement Python 3.12 dédié + AMD Quark,
ou venv séparé pour le toolchain Ryzen AI. Hors du venv sidecar actuel (py3.14).

---

## Mise à jour — Frontière SWA validée ✅ 2026-06-27

Le consensus P3 avait flaggé : *« WikiText-2 ne stresse jamais la frontière de la
sliding window (2048) → un bug de masque SWA passe invisible. »*

**Test** : séquence de **2100 tokens** (> sliding_window 2048), comparaison des
logits PyTorch patché (FP32, full forward) vs ONNX full-context, sur les 50
dernières positions (toutes post-frontière, exerçant SWA couches 0-9 +
full-attention 10-14).

| Métrique | Résultat |
|---|---|
| max abs diff logits | 2.15e-5 |
| mean abs diff | 2.32e-6 |
| top-1 match (50 pos > 2048) | **100%** |

→ **La frontière SWA est correctement exportée.** Risque #3 du spec levé.
