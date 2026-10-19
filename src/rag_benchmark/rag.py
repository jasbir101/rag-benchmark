"""RAG mode: load a corpus and a question set, chunk, embed, and build a dataset."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from rag_benchmark.chunking import Chunk, chunk_documents
from rag_benchmark.datasets import VectorDataset
from rag_benchmark.embedders import Embedder, EmbeddingCache

logger = logging.getLogger(__name__)

CORPUS_SUFFIXES = (".txt", ".md")


@dataclass(frozen=True)
class Document:
    doc_id: str  # path relative to the corpus root, with forward slashes
    text: str


@dataclass(frozen=True)
class Question:
    question: str
    answer_doc: str  # resolved to a Document.doc_id


@dataclass(frozen=True)
class RelevanceLabels:
    """Which document each indexed vector belongs to, and the answer doc of each query."""

    item_labels: list[str]
    query_targets: list[str]


def load_corpus(root: str | Path) -> list[Document]:
    """Every ``.txt`` / ``.md`` file under ``root`` (recursively), sorted by path."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    docs = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in CORPUS_SUFFIXES:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                docs.append(Document(doc_id=path.relative_to(root).as_posix(), text=text))
    if not docs:
        raise ValueError(f"no non-empty .txt or .md files under {root}")
    return docs


def _resolve_doc(name: str, docs: list[Document]) -> str | None:
    """Match ``answer_doc`` by relative path, then file name, then file stem."""
    name = name.replace("\\", "/").strip()
    ids = [d.doc_id for d in docs]
    if name in ids:
        return name
    for key in (lambda i: i.rsplit("/", 1)[-1], lambda i: Path(i).stem):
        matches = [i for i in ids if key(i) == name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"answer_doc {name!r} is ambiguous: {matches}")
    return None


def load_questions(path: str | Path, docs: list[Document]) -> list[Question]:
    """Read a JSONL file of ``{"question": ..., "answer_doc": ...}`` objects."""
    questions = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                question, answer_doc = str(record["question"]), str(record["answer_doc"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: expected question and answer_doc") from exc
            doc_id = _resolve_doc(answer_doc, docs)
            if doc_id is None:
                raise ValueError(f"{path}:{line_no}: answer_doc {answer_doc!r} not in corpus")
            questions.append(Question(question=question, answer_doc=doc_id))
    if not questions:
        raise ValueError(f"no questions in {path}")
    return questions


def build_rag_dataset(
    docs: list[Document],
    questions: list[Question],
    chunker: str,
    embedder: Embedder,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    cache: EmbeddingCache | None = None,
    corpus_name: str = "corpus",
) -> tuple[VectorDataset, RelevanceLabels, list[Chunk]]:
    """Chunk and embed the corpus and questions into a cosine :class:`VectorDataset`."""
    chunks = chunk_documents(docs, chunker, chunk_size, chunk_overlap)
    cache = cache or EmbeddingCache(None)
    logger.info("embedding %d chunks and %d questions", len(chunks), len(questions))
    base = cache.embed(embedder, [c.text for c in chunks], "documents")
    queries = cache.embed(embedder, [q.question for q in questions], "queries")
    dataset = VectorDataset(
        name=f"{corpus_name}: {len(docs)} docs, {len(chunks)} {chunker} chunks",
        base=base,
        queries=queries,
        metric="cosine",
    )
    labels = RelevanceLabels(
        item_labels=[c.doc_id for c in chunks],
        query_targets=[q.answer_doc for q in questions],
    )
    return dataset, labels, chunks
