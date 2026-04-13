# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
BGE-small singleton encoder for open-ended TTRL.

The encoder is loaded once per process and reused. We expose:
  - encode(texts)         -> np.ndarray [N, d]  (L2-normalized)
  - encode_cached(texts)  -> np.ndarray [N, d]  (uses module-level text->vec cache)
  - get_cached(text)      -> np.ndarray [d] or None
  - put_cached(text, vec) -> None
  - clear_cache()
  - cosine(u, v)          -> float

Design notes:
  * The vote module batch-encodes all rollouts at vote time and stores their
    embeddings in the cache. The reward function then performs free lookups
    when grading individual rollouts against the medoid.
  * Encoding uses CLS pooling with L2 normalization (BGE official recipe).
  * The model loads from BGE_MODEL_PATH if set, otherwise from a hardcoded
    local path under /data/sls/scratch/mvideet/models, otherwise from the
    HuggingFace hub.
  * The encoder runs on CUDA if available, falling back to CPU. With BGE-small
    (~33M params) CPU is fast enough for sanity runs.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
from typing import List, Optional

import numpy as np

_MODEL = None
_TOKENIZER = None
_DEVICE = None
_LOCK = threading.Lock()

# text-hash -> np.ndarray (float32, L2-normalized) cache
# bounded by _CACHE_MAX entries; oldest entries dropped on overflow
_CACHE: "dict[str, np.ndarray]" = {}
_CACHE_INSERTION: "list[str]" = []
_CACHE_MAX = int(os.environ.get("TTRL_OE_CACHE_MAX", "8192"))

_DEFAULT_MODEL_PATHS = [
    os.environ.get("BGE_MODEL_PATH", ""),
    "/data/sls/scratch/mvideet/models/bge-small-en-v1.5",
    "BAAI/bge-small-en-v1.5",
]


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def _load_model():
    """Lazily load BGE-small. Thread-safe."""
    global _MODEL, _TOKENIZER, _DEVICE
    if _MODEL is not None:
        return
    with _LOCK:
        if _MODEL is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        last_err = None
        chosen_path = None
        for path in _DEFAULT_MODEL_PATHS:
            if not path:
                continue
            try:
                tok = AutoTokenizer.from_pretrained(path)
                mdl = AutoModel.from_pretrained(path)
                chosen_path = path
                break
            except Exception as e:  # pragma: no cover
                last_err = e
                continue
        else:
            raise RuntimeError(
                f"[ttrl_open_ended.embedding] Could not load BGE-small from any of {_DEFAULT_MODEL_PATHS}; "
                f"last error: {last_err!r}"
            )

        # Default to CPU to avoid competing with the actor for GPU memory.
        # The encoder runs in the gap between rollout generation and actor
        # update; per-step encoding is small (N_rollouts * batch ~= 64 texts)
        # and BGE-small is fast on CPU. Override with TTRL_OE_DEVICE=cuda
        # if you need the speed boost and have headroom.
        device_pref = os.environ.get("TTRL_OE_DEVICE", "cpu").lower()
        if device_pref == "cuda" and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        mdl = mdl.to(device)
        mdl.eval()
        # Disable grad for the entire encoder.
        for p in mdl.parameters():
            p.requires_grad = False

        _MODEL = mdl
        _TOKENIZER = tok
        _DEVICE = device
        print(
            f"[ttrl_open_ended.embedding] Loaded BGE encoder from {chosen_path} on {device}",
            file=sys.stderr,
            flush=True,
        )


def _put_cache(text: str, vec: np.ndarray) -> None:
    h = _hash_text(text)
    if h in _CACHE:
        return
    _CACHE[h] = vec
    _CACHE_INSERTION.append(h)
    if len(_CACHE_INSERTION) > _CACHE_MAX:
        # Drop oldest 10% in one shot to keep amortized cost low.
        drop = max(1, _CACHE_MAX // 10)
        for _ in range(drop):
            old = _CACHE_INSERTION.pop(0)
            _CACHE.pop(old, None)


def get_cached(text: str) -> Optional[np.ndarray]:
    if text is None:
        return None
    return _CACHE.get(_hash_text(text))


def put_cached(text: str, vec: np.ndarray) -> None:
    if text is None or vec is None:
        return
    _put_cache(text, vec)


def clear_cache() -> None:
    _CACHE.clear()
    _CACHE_INSERTION.clear()


def _encode_batch(texts: List[str]) -> np.ndarray:
    """Run BGE forward on a list of texts; returns [N, d] L2-normalized float32."""
    import torch

    _load_model()
    if not texts:
        return np.zeros((0, _MODEL.config.hidden_size), dtype=np.float32)

    # BGE recommends prefixing queries with "Represent this sentence for retrieval:" but
    # for symmetric similarity (rollout vs rollout) we treat both sides as passages and
    # use the raw text. This matches the standard sentence-transformer usage of bge-small.
    enc = _TOKENIZER(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    enc = {k: v.to(_DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        out = _MODEL(**enc)
        # CLS pooling: first token of last_hidden_state.
        cls = out.last_hidden_state[:, 0]
        # L2 normalize.
        cls = torch.nn.functional.normalize(cls, p=2, dim=1)

    arr = cls.detach().to(torch.float32).cpu().numpy()
    return arr


def encode(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Encode texts in chunks of `batch_size`. Returns [N, d] float32, L2-normalized."""
    if not texts:
        _load_model()
        return np.zeros((0, _MODEL.config.hidden_size), dtype=np.float32)
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        out.append(_encode_batch(chunk))
    return np.concatenate(out, axis=0) if len(out) > 1 else out[0]


def encode_cached(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Encode texts using the module-level cache for repeated strings.

    Misses are batched and encoded in one pass; hits are reused. Result preserves
    input order.
    """
    if not texts:
        _load_model()
        return np.zeros((0, _MODEL.config.hidden_size), dtype=np.float32)

    n = len(texts)
    out: List[Optional[np.ndarray]] = [None] * n
    miss_idx: List[int] = []
    miss_text: List[str] = []
    for i, t in enumerate(texts):
        cached = get_cached(t)
        if cached is not None:
            out[i] = cached
        else:
            miss_idx.append(i)
            miss_text.append(t)

    if miss_text:
        encoded = encode(miss_text, batch_size=batch_size)
        for j, idx in enumerate(miss_idx):
            vec = encoded[j]
            out[idx] = vec
            put_cached(miss_text[j], vec)

    return np.stack(out, axis=0)


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity assuming inputs may or may not be normalized."""
    u = np.asarray(u, dtype=np.float32).reshape(-1)
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def cosine_matrix(E: np.ndarray) -> np.ndarray:
    """N x N cosine similarity matrix. Assumes E is L2-normalized."""
    E = np.asarray(E, dtype=np.float32)
    return E @ E.T
