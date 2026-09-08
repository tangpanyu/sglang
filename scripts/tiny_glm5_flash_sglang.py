#!/usr/bin/env python3
"""Run the tiny GLM-5.3-Flash adapter through SGLang on a CUDA device.

The model directory only needs the ``config.json`` emitted by
``tiny_glm5_flash.py --save-dir``.  ``--load-format dummy`` makes SGLang
random-initialize the adapter, so no original GLM checkpoint is read.
"""

from __future__ import annotations

import argparse
import os

# Keep tiny/reference runs on eager paths for the mHC and DSA pieces.  These
# defaults do not affect normal SGLang processes unless this smoke script is
# invoked.
os.environ.setdefault("SGLANG_DISABLE_DSA_INDEXER_FUSION", "1")
os.environ.setdefault("SGLANG_DSA_FUSE_TOPK", "0")
os.environ.setdefault("SGLANG_FP8_PAGED_MQA_LOGITS_TORCH", "1")
os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_PRE", "0")
os.environ.setdefault("SGLANG_OPT_DEEPGEMM_HC_PRENORM", "0")
os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_POST", "0")

import sglang as sgl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=66)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--mem-fraction-static", type=float, default=0.2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.seq_len < 1:
        raise ValueError("--batch-size and --seq-len must be positive")

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
        dsa_prefill_backend="flashinfer_sparse_mla",
        dsa_decode_backend="flashinfer_sparse_mla",
        dsa_topk_backend="torch",
        kv_cache_dtype="fp8_e4m3",
        page_size=64,
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
            sampling_params={"temperature": 0.0, "max_new_tokens": 1},
        )
        print(f"input_ids shape: [{args.batch_size}, {args.seq_len}]")
        print(f"generated batch size: {len(outputs)}")
        print("SGLang CUDA forward: ok")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
