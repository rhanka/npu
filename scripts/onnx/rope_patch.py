"""
Patch RoPE complexe → réel (sin/cos) pour rendre Laneformer exportable en ONNX.

Le RoPE de référence : freqs_cis = polar(1, angles) = cos+i·sin (complex64),
puis (x0+i·x1)·(cos+i·sin). ONNX ne supporte pas le dtype complexe.
Équivalent réel strict :  o0 = x0·cos − x1·sin ;  o1 = x0·sin + x1·cos.

On garde le MÊME rang de tenseur (angles réels au lieu de complexes) pour ne
pas casser le .unsqueeze(-2) du forward, et on réutilise compute_rope_freqs
du modèle → équivalence numérique exacte.
"""
import types
import torch


def patch_model_for_onnx(model):
    modeling = __import__(type(model).__module__, fromlist=["dummy"])
    inner = model.model

    rope_dim = inner.rope_dim
    rope_theta = inner.rope_theta
    rope_scaling_args = inner.rope_scaling_args

    def _compute_freqs_cis_real(self, position_ids):
        freqs = modeling.compute_rope_freqs(
            rope_dim, rope_theta, rope_scaling_args, device=position_ids.device,
        )
        angles = position_ids.float().unsqueeze(-1) * freqs  # [.., seq, half] réel
        return angles

    def apply_rotary_emb_real(xq, xk, freqs_cis):
        # freqs_cis = angles réels, shape [.., seq, 1, half]
        cos = torch.cos(freqs_cis)
        sin = torch.sin(freqs_cis)

        def rotate(x):
            x_ = x.float().reshape(*x.shape[:-1], -1, 2)
            x0, x1 = x_[..., 0], x_[..., 1]
            o0 = x0 * cos - x1 * sin
            o1 = x0 * sin + x1 * cos
            return torch.stack([o0, o1], dim=-1).flatten(-2)

        return rotate(xq).type_as(xq), rotate(xk).type_as(xk)

    modeling.apply_rotary_emb = apply_rotary_emb_real
    inner._compute_freqs_cis = types.MethodType(_compute_freqs_cis_real, inner)

    # Force l'attention eager (matmul/softmax explicites) : pas de SDPA is_causal
    # SymBool lors de l'export à shapes dynamiques. Le modèle custom n'honore pas
    # attn_implementation à l'init → on le force sur config + sous-modules.
    model.config._attn_implementation = "eager"
    for m in model.modules():
        if hasattr(m, "_attn_implementation"):
            m._attn_implementation = "eager"
        sub = getattr(m, "config", None)
        if sub is not None and hasattr(sub, "_attn_implementation"):
            sub._attn_implementation = "eager"
    return model
