"""Configuration for the small GLM-5.3-Flash structure adapter.

This is intentionally a model-level configuration.  It reuses the Kimi
hybrid configuration fields for KDA cache sizing and adds the GLM-5 fields for
the DSA indexer, mHC streams, and per-layer schedules.
"""

from __future__ import annotations

from typing import List, Optional

from transformers import AutoConfig

from sglang.srt.configs.kimi_linear import KimiLinearConfig


def _default_layer_types(num_hidden_layers: int) -> List[str]:
    return [
        "linear_attention" if layer_idx % 4 != 3 else "deepseek_sparse_attention"
        for layer_idx in range(num_hidden_layers)
    ]


def _default_mlp_layer_types(num_hidden_layers: int) -> List[str]:
    return ["dense"] * min(3, num_hidden_layers) + [
        "sparse"
    ] * max(0, num_hidden_layers - 3)


class Glm5FlashTinyConfig(KimiLinearConfig):
    """Scaled GLM-5.3-Flash config consumed by the SGLang adapter.

    The dimensions are small enough for a single GPU smoke test, while the
    layer schedule and the attention/MoE/mHC configuration names remain
    compatible with the existing SGLang components.
    """

    model_type = "glm5_flash_tiny"

    def __init__(
        self,
        vocab_size: int = 4096,
        hidden_size: int = 256,
        intermediate_size: int = 512,
        moe_intermediate_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        num_key_value_heads: Optional[int] = None,
        head_dim: int = 64,
        q_lora_rank: int = 64,
        # DSA's FP8 MLA cache stores the latent non-rotary KV slice in a
        # 512-wide tile (also divisible by the 128-value quantization block).
        # This cache-facing width is retained while layers/heads are tiny.
        kv_lora_rank: int = 512,
        # DSA's tiny adapter keeps the cache-facing non-rotary key slice at
        # 512-wide while scaling heads, layers, and latent ranks.
        qk_nope_head_dim: int = 512,
        # GLM-5.3-Flash DSA is no-RoPE.  Keep this zero in the adapter so the
        # serialized config and SGLang MLA path match the real model.
        qk_rope_head_dim: int = 0,
        v_head_dim: int = 64,
        linear_num_heads: int = 4,
        linear_head_dim: int = 64,
        short_conv_kernel_size: int = 4,
        linear_lower_bound: Optional[float] = -5.0,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 2,
        routed_scaling_factor: float = 2.5,
        index_topk: int = 8,
        # SGLang's DSA KV/index pool intentionally keeps the production
        # index-key width of 128 even for this tiny model.
        index_head_dim: int = 128,
        # DeepGEMM's paged-MQA indexer accepts 8/16/32/64 query heads; keep
        # the smallest supported tiny shape rather than bypassing the indexer.
        index_n_heads: int = 8,
        index_kpool: int = 4,
        index_kpool_compress: bool = True,
        hc_mult: int = 4,
        hc_eps: float = 1e-6,
        hc_sinkhorn_iters: int = 20,
        layer_types: Optional[List[str]] = None,
        mlp_layer_types: Optional[List[str]] = None,
        indexer_types: Optional[List[str]] = None,
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        # ``PretrainedConfig.from_pretrained`` feeds these serialized fields
        # back through ``kwargs``.  Consume them here because this tiny config
        # fixes both values explicitly below.
        kwargs.pop("model_type", None)
        architectures = kwargs.pop(
            "architectures", ["Glm5FlashTinyForCausalLM"]
        )
        # KimiLinearConfig serializes both its canonical fields and several
        # DeepSeek-compatible aliases.  This adapter rebuilds the canonical
        # values from the tiny arguments, so remove the duplicate aliases
        # before forwarding ``kwargs`` to the base initializer.
        for serialized_alias in (
            "first_k_dense_replace",
            "hidden_act",
            "linear_attn_config",
            "moe_layer_freq",
            "num_expert_group",
            "num_experts",
            "num_experts_per_token",
            "num_shared_experts",
            "tie_word_embeddings",
            "topk_group",
            "topk_method",
        ):
            kwargs.pop(serialized_alias, None)
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        layer_types = list(layer_types or _default_layer_types(num_hidden_layers))
        mlp_layer_types = list(
            mlp_layer_types or _default_mlp_layer_types(num_hidden_layers)
        )
        indexer_types = list(indexer_types or ["full"] * num_hidden_layers)
        if len(layer_types) != num_hidden_layers:
            raise ValueError("layer_types must match num_hidden_layers")
        if len(mlp_layer_types) != num_hidden_layers:
            raise ValueError("mlp_layer_types must match num_hidden_layers")
        if len(indexer_types) != num_hidden_layers:
            raise ValueError("indexer_types must match num_hidden_layers")
        if index_topk % index_kpool != 0:
            raise ValueError("index_topk must be divisible by index_kpool")
        if num_experts_per_tok > n_routed_experts:
            raise ValueError("num_experts_per_tok cannot exceed n_routed_experts")

        kda_layers = [
            layer_idx + 1
            for layer_idx, layer_type in enumerate(layer_types)
            if layer_type == "linear_attention"
        ]
        full_attention_layers = [
            layer_idx + 1
            for layer_idx, layer_type in enumerate(layer_types)
            if layer_type == "deepseek_sparse_attention"
        ]

        super().__init__(
            model_type=self.model_type,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act="silu",
            rms_norm_eps=rms_norm_eps,
            initializer_range=initializer_range,
            max_position_embeddings=max_position_embeddings,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=n_routed_experts,
            num_experts_per_token=num_experts_per_tok,
            num_shared_experts=n_shared_experts,
            routed_scaling_factor=routed_scaling_factor,
            first_k_dense_replace=min(3, num_hidden_layers),
            moe_layer_freq=1,
            num_expert_group=1,
            topk_group=1,
            topk_method="noaux_tc",
            linear_attn_config={
                "num_heads": linear_num_heads,
                "head_dim": linear_head_dim,
                "short_conv_kernel_size": short_conv_kernel_size,
                "gate_lower_bound": linear_lower_bound,
                "kda_layers": kda_layers,
                "full_attn_layers": full_attention_layers,
            },
            architectures=architectures,
            tie_word_embeddings=False,
            **kwargs,
        )

        # Fields consumed by DeepSeek MLA/DSA and mHC code paths.
        self.layer_types = layer_types
        self.mlp_layer_types = mlp_layer_types
        self.indexer_types = indexer_types
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_kpool = index_kpool
        self.index_kpool_compress = index_kpool_compress
        self.index_kpool_always_select_tail = True
        self.indexer_rope_interleave = True
        self.index_topk_freq = 1
        self.index_skip_topk_offset = None
        self.n_group = 1
        self.topk_group = 1
        self.norm_topk_prob = True
        self.scoring_func = "sigmoid"
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts_per_token = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_limit = 10.0
        self.linear_num_heads = linear_num_heads
        self.linear_head_dim = linear_head_dim
        self.linear_conv_kernel_dim = short_conv_kernel_size
        self.linear_lower_bound = linear_lower_bound
        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.num_hash_layers = 0


try:
    AutoConfig.register(Glm5FlashTinyConfig.model_type, Glm5FlashTinyConfig)
except (ValueError, ImportError):
    # Importing the SGLang registry more than once must remain harmless.
    pass
