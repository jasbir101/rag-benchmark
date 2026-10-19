import re
from itertools import pairwise
from types import SimpleNamespace

import pytest

from rag_benchmark.chunking import chunk_documents, fixed_size_chunks, recursive_chunks

TEXT = (
    "Tide pools form on rocky shores. Their inhabitants cope with rapid change.\n\n"
    "Sea stars are predators. Removing them lets mussels crowd out other species. "
    "That is why the sea star is called a keystone species.\n\n"
    "Visit at low tide and step only on bare rock."
)


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text)


class TestFixedSize:
    def test_respects_size(self):
        chunks = fixed_size_chunks(TEXT, size=60, overlap=10)
        assert len(chunks) > 1
        assert all(len(c) <= 60 for c in chunks)

    def test_every_word_is_covered(self):
        joined = " ".join(fixed_size_chunks(TEXT, size=50, overlap=10))
        for word in _words(TEXT):
            assert word in joined

    def test_consecutive_chunks_overlap(self):
        chunks = fixed_size_chunks("abcdefghij" * 10, size=30, overlap=10)
        for left, right in pairwise(chunks):
            assert left[-10:] == right[:10]

    def test_no_overlap(self):
        text = "x" * 100
        assert fixed_size_chunks(text, size=25, overlap=0) == ["x" * 25] * 4

    def test_prefers_word_boundaries(self):
        chunks = fixed_size_chunks("alpha beta gamma delta epsilon zeta", size=14, overlap=0)
        assert all(not c.startswith(" ") and not c.endswith(" ") for c in chunks)
        assert chunks[0] == "alpha beta"

    def test_short_and_empty_text(self):
        assert fixed_size_chunks("short", size=100, overlap=10) == ["short"]
        assert fixed_size_chunks("   ", size=100, overlap=10) == []

    @pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
    def test_invalid_parameters(self, size, overlap):
        with pytest.raises(ValueError):
            fixed_size_chunks(TEXT, size=size, overlap=overlap)


class TestRecursive:
    def test_keeps_paragraphs_that_fit(self):
        chunks = recursive_chunks(TEXT, size=200, overlap=0)
        assert chunks[0].startswith("Tide pools form on rocky shores.")
        assert all(len(c) <= 200 for c in chunks)

    def test_splits_long_paragraph_on_sentences(self):
        chunks = recursive_chunks(TEXT, size=80, overlap=0)
        assert all(len(c) <= 80 for c in chunks)
        # every chunk ends at a sentence boundary for this text
        assert all(c.endswith(".") for c in chunks)

    def test_falls_back_to_words_then_characters(self):
        words = recursive_chunks("word " * 40, size=22, overlap=0)
        assert all(len(c) <= 22 for c in words)
        assert all(set(c.split()) == {"word"} for c in words)
        chars = recursive_chunks("x" * 50, size=20, overlap=0)
        assert chars == ["x" * 20, "x" * 20, "x" * 10]

    def test_overlap_carries_trailing_pieces(self):
        text = "One. Two. Three. Four. Five. Six."
        chunks = recursive_chunks(text, size=12, overlap=5)
        assert chunks[0] == "One. Two."
        assert chunks[1].startswith("Two.")

    def test_every_word_is_covered(self):
        joined = " ".join(recursive_chunks(TEXT, size=70, overlap=0))
        assert _words(joined) == _words(TEXT)


def test_chunk_documents_labels_chunks():
    docs = [SimpleNamespace(doc_id="a.md", text=TEXT), SimpleNamespace(doc_id="b.md", text="Hi.")]
    chunks = chunk_documents(docs, "recursive", size=100, overlap=0)
    assert {c.doc_id for c in chunks} == {"a.md", "b.md"}
    a_chunks = [c for c in chunks if c.doc_id == "a.md"]
    assert [c.index for c in a_chunks] == list(range(len(a_chunks)))


def test_unknown_chunker():
    with pytest.raises(ValueError, match="unknown chunker"):
        chunk_documents([], "semantic")
