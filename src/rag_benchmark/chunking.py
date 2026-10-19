"""Text chunkers: fixed-size windows with overlap, and recursive paragraph/sentence splitting."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

_WHITESPACE = (" ", "\n", "\t")
# Paragraph break -> sentence end -> any whitespace. The last resort is a hard cut.
_RECURSIVE_LEVELS = (
    re.compile(r"\n\s*\n"),
    re.compile(r"(?<=[.!?])\s+"),
    re.compile(r"\s+"),
)


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    text: str


class _HasText(Protocol):
    doc_id: str
    text: str


def _validate(size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must be in [0, size), got {overlap} for size {size}")


def fixed_size_chunks(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Character windows of at most ``size`` characters, ``overlap`` characters apart.

    A window that would cut a word is shortened to the last whitespace in its
    second half, so most chunks end on a word boundary.
    """
    _validate(size, overlap)
    text = text.strip()
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = max(text.rfind(ch, start + size // 2, end) for ch in _WHITESPACE)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _split_recursive(text: str, size: int, level: int) -> list[str]:
    if len(text) <= size:
        return [text] if text else []
    if level >= len(_RECURSIVE_LEVELS):
        return [text[i : i + size] for i in range(0, len(text), size)]
    pieces: list[str] = []
    for part in _RECURSIVE_LEVELS[level].split(text):
        part = part.strip()
        if not part:
            continue
        if len(part) <= size:
            pieces.append(part)
        else:
            pieces.extend(_split_recursive(part, size, level + 1))
    return pieces


def _merge(pieces: list[str], size: int, overlap: int, sep: str = " ") -> list[str]:
    """Greedily pack pieces into chunks <= size, carrying up to ``overlap`` chars over."""
    chunks: list[str] = []
    window: list[str] = []
    length = 0  # == len(sep.join(window))
    for piece in pieces:
        if window and length + len(sep) + len(piece) > size:
            chunks.append(sep.join(window))
            # Drop pieces from the front until the carried tail fits the overlap
            # budget and leaves room for the new piece.
            while window and (length > overlap or length + len(sep) + len(piece) > size):
                removed = window.pop(0)
                length -= len(removed) + (len(sep) if window else 0)
        length += len(piece) + (len(sep) if window else 0)
        window.append(piece)
    if window:
        chunks.append(sep.join(window))
    return chunks


def recursive_chunks(text: str, size: int = 800, overlap: int = 0) -> list[str]:
    """Split on paragraphs, then sentences, then words, then characters, and pack the
    pieces into chunks of at most ``size`` characters.

    Unlike :func:`fixed_size_chunks`, boundaries follow the document structure, so a
    chunk rarely starts or ends mid-sentence. ``overlap`` carries whole trailing
    pieces (up to that many characters) into the next chunk.
    """
    _validate(size, overlap)
    return _merge(_split_recursive(text.strip(), size, 0), size, overlap)


CHUNKERS: dict[str, Callable[[str, int, int], list[str]]] = {
    "fixed": fixed_size_chunks,
    "recursive": recursive_chunks,
}


def chunk_documents(
    docs: Iterable[_HasText], strategy: str, size: int = 800, overlap: int = 100
) -> list[Chunk]:
    """Chunk every document with the named strategy (``fixed`` or ``recursive``)."""
    try:
        chunker = CHUNKERS[strategy]
    except KeyError:
        raise ValueError(f"unknown chunker {strategy!r}; choose from {sorted(CHUNKERS)}") from None
    chunks: list[Chunk] = []
    for doc in docs:
        for i, text in enumerate(chunker(doc.text, size, overlap)):
            chunks.append(Chunk(doc_id=doc.doc_id, index=i, text=text))
    return chunks
