"""GPU SGLang adapter for the tiny GLM-5.3-Flash structure model.

This module is deliberately a thin composition layer.  KDA, MLA/DSA, mHC,
and routed/shared MoE implementations come from SGLang's existing model
modules; this file only selects the branch for each layer and provides the
GLM5 mHC stream layout.  It is intended for eager/single-GPU validation of a
small random-weight checkpoint, not for loading the production GLM-5.3-Flash
checkpoint.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable, Optional

import torch
from torch import nn

from sglang.srt.configs.glm5_flash_tiny import Glm5FlashTinyConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models import deepseek_v4 as deepseek_v4
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA,
    DeepseekV2MLP,
)
from sglang.srt.models.kimi_linear import KimiDeltaAttention
from sglang.srt.utils import BumpAllocator, add_prefix


class _TinyKDAAdapter(nn.Module):
    """Adapt SGLang's KDA call signature to the DeepSeek mHC block."""

    def __init__(
        self,
        config: Glm5FlashTinyConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.inner = KimiDeltaAttention(
            layer_idx=layer_id,
            hidden_size=config.hidden_size,
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            rms_norm_eps=config.rms_norm_eps,
            safe_gate=config.linear_lower_bound is not None,
            lower_bound=config.linear_lower_bound,
            shard_on_attn_tp=True,
            v_head_dim=config.v_head_dim,
        )

    def maybe_use_decode_attn_tp(self, forward_batch: ForwardBatch):
        # KDA already uses the attention-TP group internally.  The context
        # manager is required by the common mHC decoder-layer call site.
        del forward_batch
        return nullcontext()

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        zero_allocator = BumpAllocator(
            buffer_size=2,
            dtype=torch.float32,
            device=x.device,
        )
        return self.inner(
            hidden_states=x,
            positions=positions,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )


class _TinyDSAAdapter(nn.Module):
    """Adapt the existing MLA+DSA module to the mHC block interface."""

    def __init__(
        self,
        config: Glm5FlashTinyConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.inner = DeepseekV2AttentionMLA(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=prefix,
            skip_rope=True,
        )
        # The production fused-A GEMMs require K dimensions such as 1024;
        # this tiny adapter deliberately uses K=256, so force the ordinary
        # eager linear projections while preserving the MLA/DSA structure.
        self.inner._use_min_latency_fused_a_gemm = False
        self.inner._use_min_latency_q_b_gemm = False

    def maybe_use_decode_attn_tp(self, forward_batch: ForwardBatch):
        return self.inner.maybe_use_decode_attn_tp(forward_batch)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        prev_topk_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        zero_allocator = BumpAllocator(
            buffer_size=2,
            dtype=torch.float32,
            device=x.device,
        )
        # DeepSeek-V2's MLA+DSA implementation obtains the low-rank q/kv
        # latent through the communicator.  DeepSeek-V4 normally installs
        # this object in its LayerCommunicator; this tiny adapter calls the
        # V4 mHC layer directly, so install the same per-call input explicitly.
        attn_context = get_attn_tp_context()
        attn_context.set_attn_inputs(
            AttentionInputs(x, forward_batch, self.inner.prepare_qkv_latent)
        )
        try:
            output = self.inner(
                positions=positions,
                hidden_states=x,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
                prev_topk_indices=prev_topk_indices,
            )
        finally:
            attn_context.clear_attn_inputs()
        # DSA eager/SDPA paths may return (attention_output, topk_indices).
        return output[0] if isinstance(output, tuple) else output


class _TinyDenseMLP(DeepseekV2MLP):
    """Dense MLP with the extra arguments used by the mHC MoE call site."""

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        return super().forward(hidden_states)


class Glm5FlashTinyDecoderLayer(deepseek_v4.DeepseekV4DecoderLayer):
    """DeepSeek mHC block whose attention branch follows GLM5's schedule."""

    def _build_self_attn(
        self,
        *,
        config: Glm5FlashTinyConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        alt_streams,
        compress_ratio_override,
    ) -> nn.Module:
        del alt_streams, compress_ratio_override
        layer_type = config.layer_types[layer_id]
        if layer_type == "linear_attention":
            return _TinyKDAAdapter(config, layer_id, quant_config, prefix)
        if layer_type == "deepseek_sparse_attention":
            return _TinyDSAAdapter(config, layer_id, quant_config, prefix)
        raise ValueError(f"Unsupported GLM5 layer type: {layer_type}")

    def __init__(
        self,
        config: Glm5FlashTinyConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams=None,
    ) -> None:
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_streams=alt_streams,
        )
        if config.mlp_layer_types[layer_id] == "dense":
            self.mlp = _TinyDenseMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                swiglu_limit=config.swiglu_limit,
            )
        elif config.mlp_layer_types[layer_id] != "sparse":
            raise ValueError(
                f"Unsupported GLM5 MLP type: {config.mlp_layer_types[layer_id]}"
            )


class Glm5FlashTinyModel(deepseek_v4.DeepseekV4Model):
    """SGLang model backbone with the GLM5 layer schedule and mHC head."""

    def __init__(
        self,
        config: Glm5FlashTinyConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.pp_group = deepseek_v4.get_pp_group()
        self.hidden_size = config.hidden_size
        if self.pp_group.is_first_rank:
            self.embed_tokens = deepseek_v4.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not deepseek_v4.is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = deepseek_v4.PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = deepseek_v4.make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Glm5FlashTinyDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = deepseek_v4.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.norm = deepseek_v4.PPMissingLayer()

        self.gemm_output_zero_allocator_size = 0
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        # GLM5 uses an unweighted final stream mean rather than DeepSeek-V4's
        # learned hyper-head.  The inherited model forward still passes these
        # arguments, so keep them as non-parameters for interface compatibility.
        self.hc_head_fn = None
        self.hc_head_base = None
        self.hc_head_scale = None
        self.dsa_enable_prefill_cp = False
        self.use_fused_mhc_post_pre = False
        self.dspark_layers_to_capture = None

    def _can_run_tbo(self, forward_batch: ForwardBatch) -> bool:
        del forward_batch
        return False

    def hc_head(self, hidden_states: torch.Tensor, *args) -> torch.Tensor:
        del args
        return hidden_states.mean(dim=1)


class Glm5FlashTinyForCausalLM(deepseek_v4.DeepseekV4ForCausalLM):
    """SGLang entry point for a random-weight tiny GLM5 GPU smoke model."""

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        del hf_config, quant_config
        # Keep the explicit shared_experts module so the tiny checkpoint's
        # state_dict remains readable and mirrors the reference model.
        return "tiny reference keeps shared_experts unfused"

    def __init__(
        self,
        config: Glm5FlashTinyConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.model = Glm5FlashTinyModel(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.pp_group = deepseek_v4.get_pp_group()
        if self.pp_group.is_last_rank:
            self.lm_head = deepseek_v4.ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=deepseek_v4.get_parallel().enable_dp_lm_head,
            )
        else:
            self.lm_head = deepseek_v4.PPMissingLayer()
        self.logits_processor = deepseek_v4.LogitsProcessor(config)
        self.start_layer = self.model.start_layer
        self.end_layer = self.model.end_layer
        self.dsa_enable_prefill_cp = False
        deepseek_v4.get_attn_tp_context().init_context(
            config.q_lora_rank, is_dsa=True
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def post_load_weights(self, is_nextn: bool = False, weight_names=None):
        """Skip DeepSeek-V4 compressor hotfixes for the tiny eager branches.

        The inherited hook assumes every attention module is a DSV4
        compressor and accesses ``compress_ratio``/``compressor``.  This tiny
        schedule intentionally uses KDA for its first three layers and the
        regular MLA+DSA indexer for the last one, so those production-only
        conversions do not apply.  We also disable fused mHC paths above, so
        no post-load cache refresh is needed.
        """
        del is_nextn, weight_names
        # MLA's absorbed DSA path caches the split kv_b_proj weights.  The
        # production loader derives these after checkpoint loading; dummy
        # initialization has no checkpoint callback, so derive them here.
        for layer in self.model.layers:
            attn = layer.self_attn
            if not isinstance(attn, _TinyDSAAdapter):
                continue
            inner = attn.inner
            w_kc, w_vc = inner.kv_b_proj.weight.unflatten(
                0, (-1, inner.qk_nope_head_dim + inner.v_head_dim)
            ).split([inner.qk_nope_head_dim, inner.v_head_dim], dim=1)
            inner.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            inner.w_vc = w_vc.contiguous().transpose(1, 2)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        with deepseek_v4.get_attn_tp_context().maybe_input_scattered(
            forward_batch
        ):
            hidden_states = self.model(
                input_ids,
                positions,
                forward_batch,
                input_embeds,
                pp_proxy_tensors,
            )
        if not self.pp_group.is_last_rank:
            return hidden_states
        hidden_states, pre_hc_head = hidden_states
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            hidden_states_before_norm=pre_hc_head,
        )

    def load_weights(self, weights: Iterable) -> set[str]:
        """Load a state_dict produced by the tiny reference model.

        This intentionally handles exact parameter names only.  Production
        GLM5 checkpoint conversion is out of scope for the random tiny model.
        """

        params_dict = dict(self.named_parameters())
        if isinstance(weights, dict):
            weights = weights.items()
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            param = params_dict.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


EntryClass = [Glm5FlashTinyForCausalLM]
