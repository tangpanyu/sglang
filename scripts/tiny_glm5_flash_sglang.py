#!/usr/bin/env python3
"""Run the tiny GLM-5.3-Flash adapter through SGLang on a CUDA device.

The model directory only needs the ``config.json`` emitted by
``tiny_glm5_flash.py --save-dir``.  ``--load-format dummy`` makes SGLang
random-initialize the adapter, so no original GLM checkpoint is read.
"""

from __future__ import annotations

import argparse
import os

import torch

# Keep tiny/reference runs on eager paths for the mHC and DSA pieces.  These
# defaults do not affect normal SGLang processes unless this smoke script is
# invoked.
os.environ.setdefault("SGLANG_DISABLE_DSA_INDEXER_FUSION", "1")
os.environ.setdefault("SGLANG_DSA_FUSE_TOPK", "0")
os.environ.setdefault("SGLANG_FP8_PAGED_MQA_LOGITS_TORCH", "1")
os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_PRE", "0")
os.environ.setdefault("SGLANG_OPT_DEEPGEMM_HC_PRENORM", "0")
os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_POST", "0")

# Register the local config before SGLang's argument-resolution pipeline calls
# ``AutoConfig.from_pretrained``.  The tiny checkpoint is intentionally a
# local debug model, so Transformers cannot discover this model_type on its
# own as it would for a published Hugging Face config.
from sglang.srt.configs.glm5_flash_tiny import Glm5FlashTinyConfig  # noqa: F401

import sglang as sgl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/tiny-glm5-flash")
    parser.add_argument("--batch-size", type=int, default=2)
    # Keep the default smoke sequence inside one DSA page.  The runtime DSA
    # backend currently requires a 64-token physical page.
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--mem-fraction-static", type=float, default=0.2)
    parser.add_argument(
        "--page-size",
        type=int,
        default=64,
        help="DSA physical page size; the current DSA backend requires 64",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.seq_len < 1:
        raise ValueError("--batch-size and --seq-len must be positive")
    if args.page_size != 64:
        raise ValueError(
            "The current DSA/FlashMLA backend requires --page-size 64"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("SGLang CUDA smoke requires a CUDA device")
    sm_major, sm_minor = torch.cuda.get_device_capability()
    if (sm_major, sm_minor) == (8, 0):
        # A100/SM80 is kept as an explicit branch: it has no native FP8 path,
        # no DeepGEMM paged-MQA logits, and no GLM SM120 FlashInfer sparse-MLA
        # kernel.  Keep both caches in BF16 and use the A100-compatible FA3
        # selected-token attention plus Torch index logits.
        dsa_prefill_backend = "fa3"
        dsa_decode_backend = "fa3"
        dsa_paged_mqa_logits_backend = "torch"
        dsa_topk_backend = "torch"
        kv_cache_dtype = "bfloat16"
    elif sm_major == 8:
        # SM86 follows the same BF16/Torch contract as A100, but remains a
        # separate branch so future SM86-specific kernels can be added without
        # changing the A100 path.
        dsa_prefill_backend = "fa3"
        dsa_decode_backend = "fa3"
        dsa_paged_mqa_logits_backend = "torch"
        dsa_topk_backend = "torch"
        kv_cache_dtype = "bfloat16"
    elif sm_major >= 9:
        dsa_prefill_backend = "flashinfer_sparse_mla"
        dsa_decode_backend = "flashinfer_sparse_mla"
        dsa_paged_mqa_logits_backend = "auto"
        dsa_topk_backend = "torch"
        kv_cache_dtype = "fp8_e4m3"
    else:
        raise RuntimeError(
            f"The tiny GLM-5.3-Flash smoke requires SM80+; got SM{sm_major}{sm_minor}."
        )

    input_ids = [
        [(batch_index * args.seq_len + token_index) % 4096 for token_index in range(args.seq_len)]
        for batch_index in range(args.batch_size)
    ]
    engine = sgl.Engine(
        model_path=args.model_dir,
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype=args.dtype,
        attention_backend="dsa",
        # Select a backend family from the active GPU.  Ampere uses BF16 MLA
        # and index caches + FA3 attention + Torch logits; SM120 keeps the FP8
        # FlashInfer path.  The tiny adapter's reduced pooled top-k shape is
        # handled by the PyTorch fallback in either indexer.
        dsa_prefill_backend=dsa_prefill_backend,
        dsa_decode_backend=dsa_decode_backend,
        dsa_paged_mqa_logits_backend=dsa_paged_mqa_logits_backend,
        dsa_topk_backend=dsa_topk_backend,
        kv_cache_dtype=kv_cache_dtype,
        page_size=args.page_size,
        # Allow the unified mamba cache to coexist with DSA's required
        # page-size=64 pool in this hybrid KDA+DSA model.
        mamba_radix_cache_strategy="extra_buffer",
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        mem_fraction_static=args.mem_fraction_static,
        random_seed=0,
    )
    try:
        outputs = engine.generate(
            input_ids=input_ids,
            sampling_params={"temperature": 0.0, "max_new_tokens": 2},
        )
        print(f"input_ids shape: [{args.batch_size}, {args.seq_len}]")
        print(f"generated batch size: {len(outputs)}")
        print("SGLang CUDA forward: ok")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
