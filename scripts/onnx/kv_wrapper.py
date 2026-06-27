"""
Wrapper KV-cache pour l'export ONNX de Laneformer 2B.

transformers 5.12.1 : DynamicCache se construit via ddp_cache_data (iterable de
tuples (key, value) par couche) et s'extrait via layers[i].keys / .values.
ONNX ne connaît pas l'objet Cache → on expose past/present comme tenseurs plats.

Le wrapper prend :
  input_ids       [B, T_new]
  position_ids    [B, T_new]
  cache_position  [T_new]
  past_k_0..N, past_v_0..N   chacun [B, n_kv, T_past, head_dim]
et renvoie :
  logits          [B, T_new, vocab]
  present_k_0..N, present_v_0..N   chacun [B, n_kv, T_past+T_new, head_dim]

Prefill = past de longueur 0. Decode = past de longueur T_past, T_new=1.
"""
import torch
import torch.nn as nn
from transformers import DynamicCache


class LaneformerKVWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layers = model.config.num_hidden_layers

    def forward(self, input_ids, position_ids, cache_position, *past):
        # past = (k0, v0, k1, v1, ...) aplati → reconstruire les paires par couche
        pairs = [(past[2 * i], past[2 * i + 1]) for i in range(self.n_layers)]
        cache = DynamicCache(ddp_cache_data=pairs)

        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        )
        present = []
        for i in range(self.n_layers):
            present.append(cache.layers[i].keys)
            present.append(cache.layers[i].values)
        return (out.logits, *present)


def empty_past(model, batch=1, dtype=torch.float32):
    """Tenseurs de cache vides [B, n_kv, 0, head_dim] pour le prefill."""
    cfg = model.config
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    past = []
    for _ in range(cfg.num_hidden_layers):
        past.append(torch.zeros(batch, n_kv, 0, head_dim, dtype=dtype))
        past.append(torch.zeros(batch, n_kv, 0, head_dim, dtype=dtype))
    return past
