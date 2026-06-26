# SPEC_STUDY — Laneformer 2B ONNX Export

**Objectif :** exporter `kogai/laneformer-2b-it` (PyTorch BF16) vers ONNX pour
permettre la quantification INT8 via AMD Quark et l'inférence sur le NPU Ryzen AI
(XDNA2 / `/dev/accel0`) ou GPU optimisé (ONNX Runtime + ROCm EP).

**Contexte hardware :** ROG Z13 — Radeon 8060S (gfx1151, ROCm 7.2) + Ryzen AI NPU
(XDNA2, /dev/accel0). Baseline actuelle : 16.7 tok/s BF16 GPU.
Cible : 80–150 tok/s NPU ou ≥40 tok/s GPU quantifié.

---

## 1. Particularités de l'architecture Laneformer

```
trust_remote_code = True
Architecture : laneformer (custom, Llama-variant)
~2.3B params, ~4.6GB BF16
Couches 0–9  : SWA (Sliding Window Attention, window 512)
Couches 10–14: Full-attention
DTP (Delayed Tensor Parallelism) : sharding logique des têtes d'attention
Tokenizer : Llama 2 (32k vocab)
Contexte : 4096 tokens
```

Les obstacles ONNX spécifiques :

| Obstacle | Sévérité | Description |
|---|---|---|
| SWA custom op | Haute | L'attention à fenêtre glissante n'est pas un op ONNX standard |
| DTP sharding | Haute | Le sharding de têtes peut produire des reshape non-tracables |
| trust_remote_code | Moyenne | Le forward pass est dans du code custom non-packagé |
| Dynamic shapes | Moyenne | seq_len variable (1 → 4096), batch_size variable |
| BF16 → FP32/FP16 | Faible | ONNX Runtime et NPU préfèrent FP32 ou FP16 ; BF16 partiel |
| KV-cache stateful | Haute | La génération token-par-token nécessite past_key_values |

---

## 2. Options d'export

### Option A — `torch.onnx.export` direct (tracing)

```python
torch.onnx.export(
    model, (input_ids, attention_mask),
    "laneformer.onnx",
    opset_version=17,
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, ...},
    do_constant_folding=True,
)
```

**Pour :** simple, aucune dépendance.
**Contre :**
- Le tracing déroule le graphe mais n'exporte pas les branches conditionnelles (SWA vs full-attention par couche).
- Les ops custom SWA seront soit inlinés (graphe très large) soit cassés.
- Pas de KV-cache dynamique sans réécrire le forward.

### Option B — HuggingFace Optimum (`optimum.exporters.onnx`)

```bash
optimum-cli export onnx --model kogai/laneformer-2b-it \
  --task text-generation-with-past laneformer_onnx/
```

**Pour :** gère `past_key_values`, dynamic shapes, opset 17+, KV-cache stateful.
**Contre :**
- `trust_remote_code` : Optimum tente de charger le modèle mais peut échouer si
  l'archi n'est pas dans son registre.
- SWA : Optimum appelle le forward PyTorch (tracing) → mêmes risques que A mais
  avec la plomberie KV-cache résolue.
- Compatibilité Laneformer non testée.

### Option C — Export manuel du forward décomposé

Réécrire un forward "ONNX-friendly" :
- Remplacer SWA par un masque d'attention statique (upper triangular + window mask)
  passé en entrée → op `Softmax` standard.
- Désactiver DTP (force single-device pour l'export).
- Exporter en mode "prefill + decode" séparé (deux graphes ONNX).

**Pour :** contrôle total, export propre, compatible NPU.
**Contre :**
- Travail significatif (~2–3 jours).
- Risque de divergence avec l'implémentation de référence.
- Nécessite validation perplexité soigneuse.

### Option D — Optimum + patch SWA custom op

Utiliser Optimum pour la structure KV-cache, puis enregistrer un custom op ONNX
pour la SWA (via `torch.onnx.register_custom_op_symbolic`).

**Pour :** meilleur des deux mondes si AMD Quark / ONNX Runtime supporte les custom ops.
**Contre :**
- ONNX Runtime / NPU Ryzen AI : les custom ops ne sont **pas supportés** sur le NPU.
  Le VitisAI EP requiert des ops purement standard.
- Utilisable pour GPU (ORT CUDA/ROCm EP) mais pas NPU.

---

## 3. Pipeline cible par backend

### GPU (ROCm EP ou CUDA EP)
```
Option B (Optimum) ou C
→ AMD Quark : AWQ 4-bit ou FP8
→ ONNX Runtime + ROCm EP
→ Cible : ~40–60 tok/s
```

### NPU (VitisAI EP / Ryzen AI)
```
Option C impérative (forward ONNX-clean, pas de custom op)
→ AMD Quark : XINT8 (INT8 poids + activations)
→ ONNX Runtime + VitisAI EP → /dev/accel0
→ Ryzen AI Software 1.7.1
→ Cible : 80–150 tok/s (si tout le modèle tient dans le NPU)
```

**Risque NPU :** le NPU Strix Halo a ~50 TOPS (INT8) mais une SRAM limitée.
Un modèle 2.3B INT8 (~2.3GB) dépasse probablement la capacité on-chip.
La stratégie réaliste : **prefill sur GPU, decode sur NPU** (les couches decode sont
les plus petites et les plus répétitives).

---

## 4. Questions ouvertes (décisions D1–D5)

**D1 — Backend cible en priorité**
GPU quantifié (plus simple, gain 2×) ou NPU (plus ambitieux, gain 5–8×) ?
Quel est le critère de succès acceptable pour une V1 ?

**D2 — Option d'export**
Option B (Optimum, rapide mais risqué sur l'archi) ou C (manuel, robuste mais lent) ?
Ou B d'abord comme tentative, avec fallback C si échec ?

**D3 — Traitement de la SWA**
Inline le masque (op standard, graphe plus grand) ou custom op (GPU uniquement) ?
Si NPU est prioritaire → inline obligatoire.

**D4 — KV-cache stateful vs. single-pass**
Export en mode génération token-par-token (2 graphes : prefill + decode, avec
past_key_values) ou export en mode "full context" (1 seul graphe, seq_len variable,
plus simple mais inutilisable pour le streaming) ?

**D5 — Validation**
Critère de régression acceptable : ΔPerplexité ≤ X% sur WikiText-2 (standard) ?
Ou test fonctionnel sur les benchmarks Kog (si disponibles) ?

---

## 5. Risques principaux

1. **SWA non-tracable** : si le code SWA utilise des boucles Python ou des
   conditionnelles sur la longueur de séquence, `torch.onnx.export` lèvera une
   erreur ou produira un graphe incorrect. → Inspecter le code source du modèle
   avant de choisir l'option.

2. **NPU capacity** : les 50 TOPS du XDNA2 sont partagés entre toutes les
   applications. Un modèle 2.3B INT8 dépasse la SRAM on-chip → latence élevée
   si tout est géré en tiled-compute. À valider avec un profil AMD Profiler.

3. **Divergence numérique** : la réécriture du forward (Option C) peut introduire
   des différences numériques subtiles. La validation perplexité est non-négociable.

4. **Ryzen AI Software version** : 1.7.1 (avril 2026) requiert Ubuntu 24.04+.
   Ubuntu 26.04 devrait être compatible mais non testé officiellement.
