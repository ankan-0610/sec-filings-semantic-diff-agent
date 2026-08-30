from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import settings

# Prefer CPU when CUDA is unavailable or incompatible with the installed PyTorch build.
# This project uses an all-MiniLM model, which runs comfortably on CPU for a watchlist workflow.
if torch.cuda.is_available():
    try:
        _device = "cuda" if torch.cuda.get_device_capability()[0] >= 7 else "cpu"
    except Exception:  # pragma: no cover - defensive fallback
        _device = "cpu"
else:
    _device = "cpu"

# Load the embedding model once at module level (required convention).
_model = SentenceTransformer(settings.EMBEDDING_MODEL, device=_device)

def chunk_by_tokens(text: str, *, window_tokens: int = 512, overlap_tokens: int = 64) -> list[str]:
    """
    Sliding-window chunking in token space, then decoding back to text.

    Note: This uses the underlying tokenizer from the sentence-transformers model.
    """
    if not text.strip():
        return []

    tokenizer = _model.tokenizer
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) <= window_tokens:
        return [text.strip()]

    chunks: list[str] = []
    start = 0
    while start < len(token_ids):
        end = min(start + window_tokens, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end == len(token_ids):
            break

        # Move forward by window - overlap.
        start = end - overlap_tokens
        if start < 0:
            start = 0

    return chunks


def embed_chunks(chunks: list[str]) -> np.ndarray:
    if not chunks:
        # Return an empty float32 vector of shape (0,) to avoid shape errors.
        return np.zeros((0,), dtype=np.float32)

    # SentenceTransformer returns numpy arrays; ensure dtype float32.
    vectors = _model.encode(chunks, convert_to_numpy=True, normalize_embeddings=False)
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors


def embed_section(text: str) -> np.ndarray:
    """
    Section-level vector = mean of all chunk vectors for that section.
    """
    chunks = chunk_by_tokens(text)
    if not chunks:
        return np.zeros((0,), dtype=np.float32)

    chunk_vectors = embed_chunks(chunks)
    # Mean over chunk dimension.
    section_vector = chunk_vectors.mean(axis=0).astype(np.float32, copy=False)
    return section_vector
