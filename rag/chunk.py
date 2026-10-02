"""Split raw document text into overlapping, citable passages.

Fixed-size character windows with overlap are used because they are simple,
deterministic and model-agnostic: the overlap keeps a sentence that straddles a
boundary retrievable from at least one chunk, and stable ids (corpus:source:ordinal)
let a citation point back to the exact passage.
"""

from __future__ import annotations

import config
from rag.types import Chunk


def chunk_text(
    text: str,
    source: str,
    corpus: str,
    size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """Cut `text` into overlapping windows of `size` characters and return Chunks.

    Each window starts `size - overlap` characters after the previous one. Windows
    that are blank after stripping are dropped (they carry no retrievable content),
    and ordinals are assigned only to kept chunks so they stay contiguous. Raises
    ValueError if `size` is not positive or `overlap` is not smaller than `size`,
    since the window would otherwise never advance.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < size, got overlap={overlap}, size={size}")

    step = size - overlap
    chunks: list[Chunk] = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{corpus}:{source}:{ordinal}",
                    text=piece,
                    source=source,
                    ordinal=ordinal,
                    corpus=corpus,
                    meta={"start": start},
                )
            )
        # The last window already reached the end of the text; stop to avoid a
        # tail chunk that is entirely contained in the previous one.
        if start + size >= len(text):
            break
    return chunks
