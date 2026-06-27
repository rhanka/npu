"""
Validation de la frontière SWA (Sliding Window Attention).

Le consensus de revue (P3) a identifié que WikiText-2 et les tests courts ne
franchissent JAMAIS la sliding_window (2048) → un bug de masque SWA passerait
invisible. Ce script génère une séquence > 2048 tokens et compare les logits
PyTorch (patché) vs ONNX aux positions post-frontière (couches SWA 0-9 vs
full-attention 10-14).

Résultat 2026-06-27 : max abs diff 2.15e-5, top-1 100% sur 50 positions >2048.
→ La frontière SWA est correctement exportée.

Usage (2 étapes, processus séparés pour la mémoire) :
    # 1. référence PyTorch → /tmp/swa_ref_logits.npy
    # 2. ONNX → comparaison
Voir l'historique git pour le détail des deux étapes.
"""
