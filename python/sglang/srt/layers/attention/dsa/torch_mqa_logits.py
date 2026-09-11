"""Small, architecture-independent DSA MQA-logits fallback.

SM80/SM86 do not provide the DeepGEMM paged-MQA kernel used by the production
DSA indexer.  Ampere stores BF16 index rows in the existing byte-backed page
facade; newer devices may still pass the FP8+scale layout.  This module decodes
either representation and evaluates the same weighted MQA dot product with
regular PyTorch operators.

This is intentionally an eager fallback.  It is useful for smoke tests and
older GPUs, not a replacement for the fused kernels on Hopper/Blackwell.
"""

from __future__ import annotations

import torch


INDEX_HEAD_DIM = 128
INDEX_ROW_BYTES = INDEX_HEAD_DIM + 4


def _as_fp8(values: torch.Tensor) -> torch.Tensor:
    if values.dtype == torch.uint8:
        return values.view(torch.float8_e4m3fn)
    return values


def _dequantize_index_rows(
    index_k: torch.Tensor, index_scale: torch.Tensor | None
) -> torch.Tensor:
    """Decode FP8 rows, or widen already-BF16 rows, into float32."""
    index_k = _as_fp8(index_k)
    if index_k.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError(
            f"expected DSA index key width {INDEX_HEAD_DIM}, got {index_k.shape}"
        )
    if index_scale is None:
        return index_k.float()
    return index_k.float() * index_scale.float().reshape(*index_scale.shape, 1)


def torch_mqa_logits(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    index_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Compute weighted MQA logits for a contiguous set of key rows.

    Args:
        q_fp8: ``[tokens, heads, 128]`` query rows (FP8 or BF16).
        weights: ``[tokens, heads]`` (or ``[..., 1]``), including query scale,
            head gate, and softmax scale.
        index_k/index_scale: cache rows and optional FP32 scales.  BF16 cache
            rows pass ``index_scale=None``.
    Returns:
        ``[tokens, key_rows]`` logits, matching the fused MQA kernels.
    """
    q = _as_fp8(q_fp8).float()
    if q.ndim != 3 or q.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError(f"expected q shape [tokens, heads, 128], got {q.shape}")
    w = weights.squeeze(-1) if weights.ndim == 3 else weights
    if w.shape != q.shape[:2]:
        raise ValueError(f"weights shape {weights.shape} does not match q {q.shape}")
    k = _dequantize_index_rows(index_k, index_scale)
    return torch.einsum("thd,kd,th->tk", q, k, w.float())


def _unpack_paged_index_cache(
    index_cache: torch.Tensor, page_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[pages, page_size, 128]`` keys and ``[pages,page_size]`` scales."""
    # The page layout is sectioned rather than row-interleaved:
    # ``[64 * fp8 key[128] | 64 * fp32 scale]``.
    raw = index_cache.view(torch.uint8).reshape(-1, page_size * INDEX_ROW_BYTES)
    key_bytes = page_size * INDEX_HEAD_DIM
    keys = raw[:, :key_bytes].reshape(-1, page_size, INDEX_HEAD_DIM)
    keys = keys.view(torch.float8_e4m3fn)
    scales = raw[:, key_bytes:].contiguous().view(torch.float32)
    return keys, scales


def torch_paged_mqa_logits(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    page_size: int,
) -> torch.Tensor:
    """Compute MQA logits from a page table without a fused paged-MQA kernel.

    ``block_tables`` maps each request to physical pages.  Rows are flattened
    in request order, which is the layout expected by DSA's top-k transform.
    Decode/verify batches may provide more than one query per request; in that
    case query rows are interpreted as ``[batch, next_n, heads, dim]``.
    """
    q = _as_fp8(q_fp8)
    if q.ndim == 3:
        q = q.unsqueeze(1)
    if q.ndim != 4:
        raise ValueError(f"expected q shape [batch,next_n,heads,128], got {q.shape}")
    bsz, next_n, heads, dim = q.shape
    if dim != INDEX_HEAD_DIM:
        raise ValueError(f"expected q head dim {INDEX_HEAD_DIM}, got {dim}")
    w = weights.squeeze(-1) if weights.ndim == 3 else weights
    if w.ndim == 2:
        w = w.unsqueeze(1)
    if w.shape != (bsz, next_n, heads):
        raise ValueError(f"weights shape {weights.shape} does not match q {q.shape}")
    if context_lens.ndim == 1:
        if context_lens.numel() == bsz:
            context_lens = context_lens[:, None].expand(bsz, next_n)
        elif context_lens.numel() == bsz * next_n:
            context_lens = context_lens.reshape(bsz, next_n)
        else:
            raise ValueError(
                f"context_lens has {context_lens.numel()} rows, expected "
                f"{bsz} or {bsz * next_n}"
            )
    elif context_lens.ndim == 2:
        if context_lens.shape == (bsz, 1):
            context_lens = context_lens.expand(bsz, next_n)
        elif context_lens.shape != (bsz, next_n):
            raise ValueError(
                f"context_lens shape {context_lens.shape} does not match "
                f"q batch/verify shape {(bsz, next_n)}"
            )
    else:
        raise ValueError(f"invalid context_lens shape {context_lens.shape}")
    context_lens = context_lens.to(torch.int64)
    if context_lens.numel() != bsz * next_n:
        raise ValueError(
            f"context_lens has {context_lens.numel()} rows, expected {bsz * next_n}"
        )
    if block_tables.ndim != 2 or block_tables.shape[0] < bsz:
        raise ValueError(f"invalid block table shape {block_tables.shape}")

    # Ampere cannot execute the FP8 index path.  Its cache uses the same raw
    # byte facade but stores BF16 key rows without a scale section.
    if index_cache.shape[-1] == page_size * INDEX_HEAD_DIM * 2:
        keys = index_cache.view(torch.bfloat16).reshape(-1, page_size, INDEX_HEAD_DIM)
        scales = None
    else:
        keys, scales = _unpack_paged_index_cache(index_cache, page_size)
    tables = block_tables[:bsz].to(torch.long)
    tables = tables.clamp(0, max(0, keys.shape[0] - 1))
    selected_k = keys[tables].reshape(bsz, -1, INDEX_HEAD_DIM)
    selected_s = scales[tables].reshape(bsz, -1) if scales is not None else None
    qf = q.float()
    kf = _dequantize_index_rows(selected_k, selected_s)
    logits = torch.einsum("bthd,bkd,bth->btk", qf, kf, w.float())
    max_len = logits.shape[-1]
    valid = torch.arange(max_len, device=logits.device)[None, None, :] < context_lens[
        :, :, None
    ]
    logits = logits.masked_fill(~valid, float("-inf"))
    return logits.reshape(bsz * next_n, max_len)
