#!/usr/bin/env python3
"""A tiny, CPU-friendly GLM-5.3-Flash *structure reference*.

The code intentionally keeps the important tensor paths of GLM-5.3-Flash while
shrinking the model dimensions, decoder depth, vocabulary, and number of
routed experts.  It is not a replacement for the production FLA/DSA kernels,
KV cache, CUDA graph, or distributed runner, and it never loads a GLM
checkpoint.  All parameters are randomly initialized.

The text-only path is::

    embedding -> mHC -> (KDA or MLA+DSA) -> mHC -> (dense or MoE) -> norm -> lm_head

The DSA branch retains the separate low-rank MLA projections and KPool indexer
path.  The MoE branch retains a learned top-k router, packed routed experts,
and an always-active shared expert.  The implementation is eager PyTorch so it
can be inspected and run on a CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _default_layer_types(num_layers: int) -> List[str]:
    """The real schedule is KDA, KDA, KDA, DSA, repeated."""

    return [
        "linear_attention" if layer_index % 4 != 3 else "deepseek_sparse_attention"
        for layer_index in range(num_layers)
    ]


def _default_mlp_layer_types(num_layers: int) -> List[str]:
    # GLM-5.3-Flash keeps the first three MLPs dense and routes the rest.
    return ["dense"] * min(3, num_layers) + ["sparse"] * max(0, num_layers - 3)


@dataclass
class TinyGlm5Config:
    """Scaled configuration with the same major fields as GLM-5.3-Flash.

    The defaults are deliberately small.  In particular, ``vocab_size``,
    ``hidden_size``, layer count, and expert count are reduced for CPU smoke
    tests; the KDA/MLA+DSA/mHC/MoE data paths are not replaced by a generic
    attention or a dense-only feed-forward block.
    """

    vocab_size: int = 4096
    hidden_size: int = 256
    num_hidden_layers: int = 4

    # Main (MLA/DSA) attention dimensions.
    num_attention_heads: int = 4
    head_dim: int = 64
    q_lora_rank: int = 64
    # GLM-5.3-Flash keeps a 512-wide latent KV tile in DSA.
    kv_lora_rank: int = 512
    # Keep the cache-facing DSA non-rotary key width used by the tiny adapter.
    qk_nope_head_dim: int = 512
    # GLM-5.3-Flash DSA is a pure-nope MLA path: it has no rotary key/query
    # slice.  The SGLang adapter must therefore run with the same zero-width
    # rope dimension instead of manufacturing a cache-only 64-wide tail.
    qk_rope_head_dim: int = 0
    v_head_dim: int = 64

    # KDA dimensions.  These are separate fields in the real configuration.
    linear_num_heads: int = 4
    linear_head_dim: int = 64
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: Optional[float] = -5.0

    intermediate_size: int = 512
    moe_intermediate_size: int = 256
    n_routed_experts: int = 4
    n_shared_experts: int = 1
    num_experts_per_tok: int = 2
    routed_scaling_factor: float = 2.5
    swiglu_limit: float = 10.0

    # DSA indexer/KPool dimensions.  index_topk is raw token positions after
    # selected pools are expanded; it must be divisible by index_kpool.
    index_topk: int = 8
    # Keep SGLang's DSA index-key width; only the number of heads is scaled.
    index_head_dim: int = 128
    # DeepGEMM's paged-MQA indexer accepts 8/16/32/64 query heads; 8 is the
    # smallest supported tiny shape while retaining the production indexer.
    index_n_heads: int = 8
    index_kpool: int = 4
    index_kpool_compress: bool = True
    index_kpool_always_select_tail: bool = True

    # Keep the four persistent mHC streams used by GLM-5.3-Flash.  Other
    # dimensions remain scaled down for the tiny CPU-friendly reference.
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    rms_norm_eps: float = 1e-6

    layer_types: List[str] = field(default_factory=lambda: _default_layer_types(4))
    mlp_layer_types: List[str] = field(
        default_factory=lambda: _default_mlp_layer_types(4)
    )
    indexer_types: List[str] = field(
        default_factory=lambda: ["full", "full", "full", "full"]
    )

    def __post_init__(self) -> None:
        # Make changing only num_hidden_layers convenient while preserving
        # explicitly supplied custom schedules.
        if len(self.layer_types) != self.num_hidden_layers:
            if self.layer_types == _default_layer_types(4):
                self.layer_types = _default_layer_types(self.num_hidden_layers)
            else:
                raise ValueError("layer_types must have num_hidden_layers entries")
        if len(self.mlp_layer_types) != self.num_hidden_layers:
            if self.mlp_layer_types == _default_mlp_layer_types(4):
                self.mlp_layer_types = _default_mlp_layer_types(self.num_hidden_layers)
            else:
                raise ValueError(
                    "mlp_layer_types must have num_hidden_layers entries"
                )
        if len(self.indexer_types) != self.num_hidden_layers:
            if all(indexer_type == "full" for indexer_type in self.indexer_types):
                self.indexer_types = ["full"] * self.num_hidden_layers
            else:
                raise ValueError("indexer_types must have num_hidden_layers entries")

        if self.num_experts_per_tok < 1 or self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError(
                "num_experts_per_tok must be in [1, n_routed_experts]"
            )
        if self.n_shared_experts < 1:
            raise ValueError("n_shared_experts must be positive")
        if self.hc_mult < 1:
            raise ValueError("hc_mult must be positive")
        if self.index_topk < self.index_kpool or self.index_topk % self.index_kpool:
            raise ValueError("index_topk must be divisible by index_kpool")
        if self.index_head_dim < 1 or self.index_n_heads < 1:
            raise ValueError("DSA index dimensions must be positive")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, object]) -> "TinyGlm5Config":
        values = dict(values)
        # Keep the standalone dataclass loader compatible with the
        # HuggingFace metadata that makes the same directory consumable by
        # SGLang.
        values.pop("model_type", None)
        values.pop("architectures", None)
        return cls(**values)


class RMSNorm(nn.Module):
    """RMSNorm with the float32 accumulation used by the reference path."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_float = hidden_states.float()
        variance = hidden_float.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_float * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)


class UnweightedRMSNorm(nn.Module):
    """The parameter-free stream normalization used inside mHC."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_float = hidden_states.float()
        variance = hidden_float.pow(2).mean(dim=-1, keepdim=True)
        return (hidden_float * torch.rsqrt(variance + self.eps)).to(hidden_states.dtype)


class RMSNormGated(nn.Module):
    """Per-head RMSNorm followed by a sigmoid gate, as in KDA output."""

    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(
        self, hidden_states: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_float = hidden_states.float()
        variance = hidden_float.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_float * torch.rsqrt(variance + self.eps)
        gated = normalized * self.weight.float() * torch.sigmoid(gate.float())
        return gated.to(input_dtype)


class TinyKDAForgetGate(nn.Module):
    """Full KDA forget-gate projection with per-head decay parameters."""

    def __init__(self, config: TinyGlm5Config) -> None:
        super().__init__()
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.qkv_dim = self.head_dim * self.num_heads
        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.qkv_dim))
        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        self.safe_gate_lower_bound = config.linear_lower_bound
        nn.init.normal_(self.dt_bias, mean=-3.0, std=0.3)
        nn.init.normal_(self.A_log, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[:2]
        raw_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        gate = (
            raw_gate.float() + self.dt_bias.float().view(1, 1, -1)
        ).view(batch_size, seq_len, self.num_heads, self.head_dim)
        decay_rate = self.A_log.float().exp().view(1, 1, self.num_heads, 1)
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * torch.sigmoid(decay_rate * gate)
        # This branch mirrors the unconstrained FLA parameterization.
        softplus = torch.where(
            gate > 20.0, gate, torch.log1p(torch.exp(gate))
        )
        return -decay_rate * softplus


class TinyKDA(nn.Module):
    """Kimi Delta Attention with the eager recurrent delta-rule update."""

    def __init__(self, config: TinyGlm5Config, layer_index: int = 0) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.layer_type = config.layer_types[layer_index]
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        self.qkv_dim = self.num_heads * self.head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

        self.q_proj = nn.Linear(config.hidden_size, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.qkv_dim, bias=False)
        self.conv1d = nn.Conv1d(
            3 * self.qkv_dim,
            3 * self.qkv_dim,
            kernel_size=self.conv_kernel_size,
            groups=3 * self.qkv_dim,
            bias=False,
        )
        self.forget_gate = TinyKDAForgetGate(config)
        self.b_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.g_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.o_norm = RMSNormGated(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(self.qkv_dim, config.hidden_size, bias=False)
        self.last_recurrent_state: Optional[torch.Tensor] = None

    def _causal_short_conv(self, mixed_qkv: torch.Tensor) -> torch.Tensor:
        # mixed_qkv is [B, 3*qkv_dim, T].  Left padding plus tail slicing is
        # equivalent to the causal_conv1d call in the production implementation.
        input_length = mixed_qkv.shape[-1]
        mixed_qkv = F.pad(mixed_qkv, (self.conv_kernel_size - 1, 0))
        mixed_qkv = self.conv1d(mixed_qkv)
        return F.silu(mixed_qkv[..., -input_length:])

    def forward(
        self,
        hidden_states: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = torch.cat(
            [
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            ],
            dim=-1,
        ).transpose(1, 2)
        mixed_qkv = self._causal_short_conv(mixed_qkv).transpose(1, 2)
        query, key, value = mixed_qkv.split([self.qkv_dim] * 3, dim=-1)
        shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        query = query.view(shape)
        key = key.view(shape)
        value = value.view(shape)

        # FLA performs this state update in float32 even when activations are
        # BF16.  The state is [B, heads, key_dim, value_dim].
        query_float = query.float()
        key_float = key.float()
        value_float = value.float()
        query_float = query_float / torch.sqrt(
            query_float.pow(2).sum(dim=-1, keepdim=True) + 1e-6
        )
        key_float = key_float / torch.sqrt(
            key_float.pow(2).sum(dim=-1, keepdim=True) + 1e-6
        )
        query_float = query_float * (self.head_dim**-0.5)
        decay = self.forget_gate(hidden_states)
        beta = torch.sigmoid(self.b_proj(hidden_states)).float()

        if initial_state is None:
            recurrent_state = torch.zeros(
                batch_size,
                self.num_heads,
                self.head_dim,
                self.head_dim,
                dtype=torch.float32,
                device=hidden_states.device,
            )
        else:
            recurrent_state = initial_state.float().to(hidden_states.device)
        core_output = torch.empty(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
            dtype=torch.float32,
            device=hidden_states.device,
        )

        for time_index in range(seq_len):
            decay_t = decay[:, time_index].float().exp().unsqueeze(-1)
            key_t = key_float[:, time_index]
            value_t = value_float[:, time_index]
            query_t = query_float[:, time_index]
            recurrent_state = recurrent_state * decay_t
            remembered_value = (recurrent_state * key_t.unsqueeze(-1)).sum(dim=-2)
            delta = (value_t - remembered_value) * beta[:, time_index].unsqueeze(-1)
            recurrent_state = recurrent_state + key_t.unsqueeze(-1) * delta.unsqueeze(-2)
            core_output[:, time_index] = (
                recurrent_state * query_t.unsqueeze(-1)
            ).sum(dim=-2)

        self.last_recurrent_state = recurrent_state.detach()
        core_output = core_output.to(hidden_states.dtype)
        output_gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(shape)
        core_output = self.o_norm(core_output, output_gate)
        output = self.o_proj(core_output.reshape(batch_size, seq_len, self.qkv_dim))
        if return_state:
            return output, recurrent_state
        return output


class TinyDSAIndexer(nn.Module):
    """GLM-5.3-Flash-style DSA indexer with KPool compression.

    The indexer first scores complete pools, expands selected pools back to raw
    token positions, and appends the visible incomplete tail.  It therefore
    preserves the important distinction between index selection and the sparse
    latent attention that consumes the selected positions.
    """

    def __init__(self, config: TinyGlm5Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        # These aliases match the names used by SGLang/Transformers indexers.
        self.n_heads = self.index_n_heads
        self.head_dim = self.index_head_dim
        self.index_topk = config.index_topk
        self.index_kpool = config.index_kpool
        self.index_kpool_always_select_tail = config.index_kpool_always_select_tail
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            bias=False,
        )
        self.wk = nn.Linear(config.hidden_size, self.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(
            config.hidden_size, self.index_n_heads, bias=False
        )
        # GLM-5.3-Flash's indexer checkpoint contains both k_norm.weight and
        # k_norm.bias.  This is LayerNorm, unlike the RMSNorm used by the
        # surrounding decoder blocks.
        self.k_norm = nn.LayerNorm(self.index_head_dim, eps=config.rms_norm_eps)
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.index_kpool, self.index_head_dim)
        )
        self.index_kpool_compress_gate = nn.Linear(
            config.hidden_size, self.index_head_dim, bias=False
        )
        self.last_topk_indices: Optional[torch.Tensor] = None

    @property
    def wq_b(self) -> nn.Linear:
        """Runtime-style alias for the indexer's query projection."""

        return self.q_b_proj

    def _pool_keys(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        keys = self.k_norm(self.wk(hidden_states))
        gates = self.index_kpool_compress_gate(hidden_states)
        num_pools = max(1, (seq_len + self.index_kpool - 1) // self.index_kpool)
        padded_len = num_pools * self.index_kpool
        pad_len = padded_len - seq_len
        if pad_len:
            keys = F.pad(keys, (0, 0, 0, pad_len))
            gates = F.pad(gates, (0, 0, 0, pad_len))
        valid = torch.cat(
            [
                torch.ones(
                    batch_size, seq_len, device=hidden_states.device, dtype=torch.bool
                ),
                torch.zeros(
                    batch_size, pad_len, device=hidden_states.device, dtype=torch.bool
                ),
            ],
            dim=1,
        )
        grouped_keys = keys.view(batch_size, num_pools, self.index_kpool, -1)
        grouped_gates = gates.view(batch_size, num_pools, self.index_kpool, -1)
        grouped_valid = valid.view(batch_size, num_pools, self.index_kpool)
        pool_logits = grouped_gates.float() + self.index_kpool_compress_ape.float()
        pool_logits = pool_logits.masked_fill(~grouped_valid[..., None], float("-inf"))
        probabilities = torch.nan_to_num(
            torch.softmax(pool_logits, dim=2), nan=0.0
        ).to(grouped_keys.dtype)
        pool_keys = (probabilities * grouped_keys).sum(dim=2)
        pool_indices = torch.arange(
            padded_len, device=hidden_states.device, dtype=torch.long
        ).view(num_pools, self.index_kpool)
        pool_complete = grouped_valid.all(dim=-1)
        return pool_keys, pool_indices, pool_complete

    def forward(
        self, hidden_states: torch.Tensor, q_residual: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        pool_keys, pool_indices, pool_complete = self._pool_keys(hidden_states)
        query = self.q_b_proj(q_residual).view(
            batch_size, seq_len, self.index_n_heads, self.index_head_dim
        )
        pool_scores = torch.einsum(
            "btid,bpd->btip", query.float(), pool_keys.float()
        )
        pool_scores = F.relu(pool_scores * (self.index_head_dim**-0.5))
        head_weights = self.weights_proj(hidden_states).float() * (
            self.index_n_heads**-0.5
        )
        index_scores = (
            pool_scores * head_weights.unsqueeze(-1)
        ).sum(dim=2)  # [B, T, pools]

        output_width = self.index_topk + (
            self.index_kpool - 1 if self.index_kpool_always_select_tail else 0
        )
        topk_indices = torch.full(
            (batch_size, seq_len, output_width),
            -1,
            dtype=torch.long,
            device=hidden_states.device,
        )
        max_pool_count = pool_indices.shape[0]
        pools_to_select = self.index_topk // self.index_kpool
        for time_index in range(seq_len):
            visible_count = time_index + 1
            complete_count = min(visible_count // self.index_kpool, max_pool_count)
            if complete_count:
                select_count = min(pools_to_select, complete_count)
                _, selected_pool_ids = torch.topk(
                    index_scores[:, time_index, :complete_count],
                    k=select_count,
                    dim=-1,
                )
                selected = pool_indices[selected_pool_ids]
                selected = selected.reshape(batch_size, -1)
                selected_width = selected.shape[-1]
                topk_indices[:, time_index, :selected_width] = selected
            else:
                selected_width = 0

            if self.index_kpool_always_select_tail:
                tail_start = complete_count * self.index_kpool
                tail_count = visible_count % self.index_kpool
                if tail_count:
                    tail = torch.arange(
                        tail_start,
                        tail_start + tail_count,
                        device=hidden_states.device,
                        dtype=torch.long,
                    )
                    topk_indices[:, time_index, selected_width : selected_width + tail_count] = tail

        # ``pool_complete`` is computed explicitly to keep the KPool structure
        # visible; complete_count above enforces the same causal validity.
        del pool_complete
        self.last_topk_indices = topk_indices.detach()
        return topk_indices


class TinyDSA(nn.Module):
    """MLA latent attention whose visible keys are selected by the DSA indexer."""

    def __init__(
        self, config: TinyGlm5Config, layer_index: int = 0
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.layer_type = config.layer_types[layer_index]
        self.indexer_type = config.indexer_types[layer_index]
        self.index_topk = config.index_topk
        self.index_kpool = config.index_kpool

        # Low-rank MLA query and compressed KV projections are retained rather
        # than replacing DSA with ordinary full q/k/v projections.
        self.q_a_proj = nn.Linear(config.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, self.kv_lora_rank, bias=False
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, config.hidden_size, bias=False
        )
        self.indexer = TinyDSAIndexer(config)
        self.last_topk_indices: Optional[torch.Tensor] = None

    def _sparse_latent_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _, _ = query.shape
        key_by_head = key.permute(0, 2, 1, 3)
        value_by_head = value.permute(0, 2, 1, 3)
        output = torch.zeros(
            batch_size,
            seq_len,
            self.num_heads,
            self.v_head_dim,
            dtype=torch.float32,
            device=query.device,
        )
        kv_len = key.shape[1]
        for time_index in range(seq_len):
            indices = topk_indices[:, time_index]
            valid = indices.ge(0) & indices.lt(kv_len)
            safe_indices = indices.clamp(0, kv_len - 1)
            gather_index = safe_indices[:, None, :, None].expand(
                batch_size, self.num_heads, safe_indices.shape[-1], self.qk_head_dim
            )
            selected_keys = torch.gather(key_by_head, 2, gather_index)
            value_index = safe_indices[:, None, :, None].expand(
                batch_size, self.num_heads, safe_indices.shape[-1], self.v_head_dim
            )
            selected_values = torch.gather(value_by_head, 2, value_index)
            scores = (
                query[:, time_index].float().unsqueeze(2) * selected_keys.float()
            ).sum(dim=-1) * (self.qk_head_dim**-0.5)
            scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            output[:, time_index] = (
                weights.unsqueeze(-1) * selected_values.float()
            ).sum(dim=2)
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        q_residual = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query = self.q_b_proj(q_residual).view(
            batch_size, seq_len, self.num_heads, self.qk_head_dim
        )
        compressed_kv = self.kv_a_layernorm(
            self.kv_a_proj_with_mqa(hidden_states)
        )
        expanded_kv = self.kv_b_proj(compressed_kv).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        key, value = torch.split(
            expanded_kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        # GLM-5.3-Flash has no rotary DSA slice.  Keep key/query widths equal
        # to qk_nope_head_dim; the SGLang path uses the same no-RoPE layout.

        if self.indexer_type == "shared":
            if prev_topk_indices is None:
                raise ValueError(
                    "A shared DSA indexer requires top-k indices from a full indexer"
                )
            topk_indices = prev_topk_indices
        elif self.indexer_type == "full":
            topk_indices = self.indexer(hidden_states, q_residual)
        else:
            raise ValueError(f"Unsupported indexer type: {self.indexer_type}")

        sparse_output = self._sparse_latent_attention(
            query, key, value, topk_indices
        )
        output = self.o_proj(
            sparse_output.to(hidden_states.dtype).reshape(
                batch_size, seq_len, self.num_heads * self.v_head_dim
            )
        )
        self.last_topk_indices = topk_indices.detach()
        return output, topk_indices


class TinySwiGLU(nn.Module):
    """GLM-5-style clamped SwiGLU dense MLP/shared expert."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, swiglu_limit: float = 10.0
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states).clamp(max=self.swiglu_limit)
        up = self.up_proj(hidden_states).clamp(
            min=-self.swiglu_limit, max=self.swiglu_limit
        )
        return self.down_proj(F.silu(gate) * up)


class TinyMoERouter(nn.Module):
    """DeepSeek-style sigmoid top-k router with correction bias."""

    def __init__(self, config: TinyGlm5Config) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_group = 1
        self.topk_group = 1
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(self.num_experts), persistent=True
        )
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(flat_states.float(), self.weight.float())
        scores = torch.sigmoid(router_logits)
        scores_for_choice = scores + self.e_score_correction_bias.float()
        experts_per_group = self.num_experts // self.num_group
        grouped = scores_for_choice.view(-1, self.num_group, experts_per_group)
        group_k = min(2, experts_per_group)
        group_scores = grouped.topk(group_k, dim=-1).values.sum(dim=-1)
        selected_groups = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        ).indices
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, selected_groups, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.num_group, experts_per_group)
            .reshape(-1, self.num_experts)
            .bool()
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask, float("-inf"))
        topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        ).indices
        topk_weights = scores.gather(1, topk_indices)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices


class TinyMoEExperts(nn.Module):
    """Packed routed expert weights, matching the runtime expert layout."""

    def __init__(self, config: TinyGlm5Config) -> None:
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                2 * self.intermediate_dim,
                self.hidden_dim,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj, mean=0.0, std=0.02)

    def forward(
        self,
        flat_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(flat_states)
        for expert_index in range(self.num_experts):
            token_index, topk_position = torch.where(topk_indices == expert_index)
            if token_index.numel() == 0:
                continue
            current = F.linear(flat_states[token_index], self.gate_up_proj[expert_index])
            gate, up = current.chunk(2, dim=-1)
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            current = F.silu(gate) * up
            current = F.linear(current, self.down_proj[expert_index])
            current = current * topk_weights[token_index, topk_position, None]
            output.index_add_(0, token_index, current.to(output.dtype))
        return output


class TinyMoE(nn.Module):
    """Routed top-k experts plus an always-active shared expert."""

    def __init__(self, config: TinyGlm5Config) -> None:
        super().__init__()
        self.config = config
        self.num_shared_experts = config.n_shared_experts
        self.gate = TinyMoERouter(config)
        self.experts = TinyMoEExperts(config)
        self.shared_experts = TinySwiGLU(
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
            config.swiglu_limit,
        )
        self.last_topk_indices: Optional[torch.Tensor] = None
        self.last_router_logits: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        residual = hidden_states
        router_logits, topk_weights, topk_indices = self.gate(hidden_states)
        flat_states = hidden_states.reshape(-1, original_shape[-1])
        routed = self.experts(flat_states, topk_indices, topk_weights).view(
            original_shape
        )
        shared = self.shared_experts(residual)
        self.last_topk_indices = topk_indices.detach()
        self.last_router_logits = router_logits.detach()
        return routed + shared


class TinyMHC(nn.Module):
    """Manifold-constrained hyper-connection (mHC) mapping.

    Inputs use the GLM/DeepSeek layout ``[B, T, hc_mult, hidden]``.  ``pre``
    collapses streams before a sublayer, while ``post`` and the Sinkhorn-
    projected ``comb`` place the sublayer result back into all streams.
    """

    def __init__(self, config: TinyGlm5Config) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.input_norm = UnweightedRMSNorm(config.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.zeros(mix))
        self.scale = nn.Parameter(torch.ones(3))
        nn.init.normal_(self.fn, mean=0.0, std=0.02)

    def forward(
        self, hidden_streams: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = self.hc_mult
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split(
            [hc, hc, hc * hc], dim=-1
        )
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2.0 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale
        comb_logits = comb_logits + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(max(0, self.hc_sinkhorn_iters - 1)):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2)
        return post, comb, collapsed.to(hidden_streams.dtype)


class TinyMHCHead(nn.Module):
    """GLM-5.3-Flash's final unweighted stream collapse."""

    def __init__(self, hc_mult: int) -> None:
        super().__init__()
        self.hc_mult = hc_mult

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=2)


class TinyMHCBlock(nn.Module):
    """One GLM-5-style decoder block with two mHC sites."""

    def __init__(
        self,
        config: TinyGlm5Config,
        layer_index: int,
        attention: nn.Module,
        mlp: nn.Module,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.block_type = config.layer_types[layer_index]
        self.mlp_layer_type = config.mlp_layer_types[layer_index]
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.attn_hc = TinyMHC(config)
        self.ffn_hc = TinyMHC(config)
        self.mhc_forward_count = 0

    @property
    def attention(self) -> nn.Module:
        """Compatibility/readability alias for the attention branch."""

        return self.self_attn

    def forward(
        self,
        hidden_streams: torch.Tensor,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self.mhc_forward_count += 1
        residual = hidden_streams
        post, comb, collapsed = self.attn_hc(hidden_streams)
        collapsed = self.input_layernorm(collapsed)
        topk_indices = None
        if self.block_type == "linear_attention":
            attention_output = self.self_attn(collapsed)
        elif self.block_type == "deepseek_sparse_attention":
            attention_output, topk_indices = self.self_attn(
                collapsed, prev_topk_indices=prev_topk_indices
            )
        else:
            raise ValueError(f"Unsupported layer type: {self.block_type}")
        hidden_streams = post.to(hidden_streams.dtype).unsqueeze(-1) * attention_output.unsqueeze(-2)
        hidden_streams = hidden_streams + torch.matmul(
            comb.to(hidden_streams.dtype).transpose(-1, -2), residual
        )

        residual = hidden_streams
        post, comb, collapsed = self.ffn_hc(hidden_streams)
        collapsed = self.post_attention_layernorm(collapsed)
        mlp_output = self.mlp(collapsed)
        hidden_streams = post.to(hidden_streams.dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2)
        hidden_streams = hidden_streams + torch.matmul(
            comb.to(hidden_streams.dtype).transpose(-1, -2), residual
        )
        return hidden_streams, topk_indices


class TinyGlm5FlashModel(nn.Module):
    """Text backbone, kept separate so a future runner adapter can wrap it."""

    def __init__(self, config: TinyGlm5Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        blocks = []
        for layer_index, (layer_type, mlp_type) in enumerate(
            zip(config.layer_types, config.mlp_layer_types)
        ):
            if layer_type == "linear_attention":
                attention = TinyKDA(config, layer_index)
            elif layer_type == "deepseek_sparse_attention":
                attention = TinyDSA(config, layer_index)
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            if mlp_type == "dense":
                mlp = TinySwiGLU(
                    config.hidden_size, config.intermediate_size, config.swiglu_limit
                )
            elif mlp_type == "sparse":
                mlp = TinyMoE(config)
            else:
                raise ValueError(f"Unsupported MLP type: {mlp_type}")
            blocks.append(TinyMHCBlock(config, layer_index, attention, mlp))
        self.layers = nn.ModuleList(blocks)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hc_head = TinyMHCHead(config.hc_mult)
        self.last_topk_indices: Optional[torch.Tensor] = None

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            if input_ids.ndim != 2:
                raise ValueError("input_ids must have shape [batch, seq_len]")
            inputs_embeds = self.embed_tokens(input_ids.long())
        hidden_streams = inputs_embeds.unsqueeze(2).expand(
            -1, -1, self.config.hc_mult, -1
        ).contiguous()
        topk_indices = None
        for layer in self.layers:
            hidden_streams, layer_topk = layer(
                hidden_streams, prev_topk_indices=topk_indices
            )
            if layer_topk is not None:
                topk_indices = layer_topk
        self.last_topk_indices = topk_indices
        return self.norm(self.hc_head(hidden_streams))


class TinyGlm5FlashForCausalLM(nn.Module):
    """Causal LM wrapper with independently sized embedding and lm_head."""

    def __init__(self, config: Optional[TinyGlm5Config] = None) -> None:
        super().__init__()
        self.config = config or TinyGlm5Config()
        self.model = TinyGlm5FlashModel(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )

    # Keep the convenient names used by the first smoke-test version without
    # registering duplicate module references in state_dict.
    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.model.embed_tokens

    @property
    def layers(self) -> nn.ModuleList:
        return self.model.layers

    @property
    def norm(self) -> RMSNorm:
        return self.model.norm

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds)
        return self.lm_head(hidden_states)


def _parse_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = dtype_name.lower()
    if dtype_name in {"float32", "fp32"}:
        return torch.float32
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype_name in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"],
    )
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    if args.seq_len < 1 or args.batch_size < 1:
        raise ValueError("--seq-len and --batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)
    config = TinyGlm5Config()
    model = TinyGlm5FlashForCausalLM(config).to(device=device, dtype=dtype)
    input_ids = torch.randint(
        0, config.vocab_size, (args.batch_size, args.seq_len), device=device
    )
    with torch.no_grad():
        logits = model(input_ids=input_ids)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"input_ids shape: {list(input_ids.shape)}")
    print(f"logits shape: {list(logits.shape)}")
    print(f"parameters: {parameter_count:,}")
    print(f"layer_types: {config.layer_types}")
    print(f"mlp_layer_types: {config.mlp_layer_types}")

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        config_dict = config.to_dict()
        # Let SGLang discover the adapter from this directory.  Its
        # ``--load-format dummy`` path intentionally initializes compatible
        # random weights instead of loading this standalone state dict.
        config_dict["model_type"] = "glm5_flash_tiny"
        config_dict["architectures"] = ["Glm5FlashTinyForCausalLM"]
        with open(
            os.path.join(args.save_dir, "config.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(config_dict, file, indent=2)
            file.write("\n")
        torch.save(
            model.state_dict(), os.path.join(args.save_dir, "pytorch_model.bin")
        )
        print(f"saved: {args.save_dir}")


if __name__ == "__main__":
    main()
