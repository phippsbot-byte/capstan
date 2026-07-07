# Local Hy3 / hy_v3 MLX canary module.
# Built for mlx-community/Hy3-preview-4bit weight names on the Studio.
# This is a runtime canary, not a production Capstan lane.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int = 4096
    num_hidden_layers: int = 80
    intermediate_size: int = 13312
    moe_intermediate_size: Optional[int] = None
    expert_hidden_dim: Optional[int] = None
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_parameters: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False

    # Hy3 MoE
    num_experts: int = 192
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    first_k_dense_replace: int = 1
    route_norm: bool = True
    router_scaling_factor: float = 2.826
    moe_router_use_sigmoid: bool = True
    moe_router_enable_expert_bias: bool = True
    qk_norm: bool = True

    # ignored compatibility knobs
    attention_bias: bool = False
    num_nextn_predict_layers: int = 0

    def __post_init__(self):
        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = self.expert_hidden_dim or self.intermediate_size
        if self.rope_parameters and self.rope_parameters.get("rope_theta"):
            self.rope_theta = float(self.rope_parameters["rope_theta"])


class RoPE(nn.Module):
    def __init__(self, dims: int, base: float):
        super().__init__()
        self.dims = dims
        self.base = base

    def __call__(self, x, offset: int = 0):
        return mx.fast.rope(
            x,
            self.dims,
            traditional=False,
            base=self.base,
            scale=1.0,
            offset=offset,
        )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or (dim // self.n_heads)
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, args.rms_norm_eps) if args.qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, args.rms_norm_eps) if args.qk_norm else None
        self.rope = RoPE(self.head_dim, args.rope_theta)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.q_norm is not None:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        offset = cache.offset if cache else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Router(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = mx.zeros((args.num_experts,), dtype=mx.float32) if args.moe_router_enable_expert_bias else None

    def __call__(self, x: mx.array) -> mx.array:
        return self.gate(x)


class Hy3MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.route_norm = args.route_norm
        self.router_scaling_factor = args.router_scaling_factor
        self.use_sigmoid = args.moe_router_use_sigmoid

        self.router = Router(args)
        self.switch_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, args.num_experts)
        if args.num_shared_experts > 0:
            self.shared_mlp = MLP(args.hidden_size, args.moe_intermediate_size * args.num_shared_experts)
        else:
            self.shared_mlp = None

    def __call__(self, x: mx.array):
        gates = self.router(x)
        if self.use_sigmoid:
            scores = mx.sigmoid(gates.astype(mx.float32))
        else:
            scores = mx.softmax(gates.astype(mx.float32), axis=-1)

        selection_scores = scores
        if self.router.expert_bias is not None:
            selection_scores = selection_scores + self.router.expert_bias

        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-selection_scores, kth=k - 1, axis=-1)[..., :k])
        selected_scores = mx.take_along_axis(scores, inds, axis=-1)
        if self.route_norm and k > 1:
            selected_scores = selected_scores / selected_scores.sum(axis=-1, keepdims=True)
        selected_scores = selected_scores * self.router_scaling_factor

        y = self.switch_mlp(x, inds)
        y = (y * selected_scores[..., None].astype(mx.float32)).sum(axis=-2).astype(y.dtype)
        if self.shared_mlp is not None:
            y = y + self.shared_mlp(x)
        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args)
        if layer_idx < args.first_k_dense_replace:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        else:
            self.mlp = Hy3MoE(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Hy3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Hy3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k
        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if "router.gate" in path:
                return {"group_size": 64, "bits": 8}
            return True
        return predicate
