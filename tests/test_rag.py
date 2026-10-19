import json

import pytest

from rag_benchmark.adapters import IndexConfig
from rag_benchmark.embedders import HashingEmbedder
from rag_benchmark.rag import build_rag_dataset, load_corpus, load_questions
from rag_benchmark.runner import run_benchmark

from .conftest import SAMPLE_DIR


@pytest.fixture
def corpus_dir(tmp_path):
    root = tmp_path / "corpus"
    (root / "garden").mkdir(parents=True)
    (root / "garden" / "compost.md").write_text(
        "Compost needs greens and browns.\n\nTurn the pile for oxygen.", encoding="utf-8"
    )
    (root / "coffee.txt").write_text("Grind coffee beans just before brewing.", encoding="utf-8")
    (root / "notes.csv").write_text("ignored,file", encoding="utf-8")
    (root / "empty.md").write_text("   ", encoding="utf-8")
    return root


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_load_corpus(corpus_dir):
    docs = load_corpus(corpus_dir)
    assert [d.doc_id for d in docs] == ["coffee.txt", "garden/compost.md"]


def test_load_corpus_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "missing")
    with pytest.raises(ValueError):
        load_corpus(tmp_path)


def test_answer_doc_resolution(corpus_dir, tmp_path):
    docs = load_corpus(corpus_dir)
    path = _write_jsonl(
        tmp_path / "q.jsonl",
        [
            {"question": "how to compost?", "answer_doc": "garden/compost.md"},
            {"question": "pile?", "answer_doc": "compost.md"},
            {"question": "beans?", "answer_doc": "coffee"},
        ],
    )
    questions = load_questions(path, docs)
    assert [q.answer_doc for q in questions] == [
        "garden/compost.md",
        "garden/compost.md",
        "coffee.txt",
    ]


def test_unknown_answer_doc_reports_line(corpus_dir, tmp_path):
    docs = load_corpus(corpus_dir)
    path = _write_jsonl(
        tmp_path / "q.jsonl",
        [{"question": "a", "answer_doc": "coffee"}, {"question": "b", "answer_doc": "tea.md"}],
    )
    with pytest.raises(ValueError, match=r"q\.jsonl:2"):
        load_questions(path, docs)


def test_malformed_question_line(corpus_dir, tmp_path):
    docs = load_corpus(corpus_dir)
    path = tmp_path / "q.jsonl"
    path.write_text('{"question": "missing answer"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="answer_doc"):
        load_questions(path, docs)


def test_build_dataset_and_score(corpus_dir, tmp_path):
    docs = load_corpus(corpus_dir)
    path = _write_jsonl(
        tmp_path / "q.jsonl",
        [
            {"question": "when should coffee beans be ground?", "answer_doc": "coffee.txt"},
            {"question": "compost greens and browns", "answer_doc": "compost.md"},
        ],
    )
    questions = load_questions(path, docs)
    dataset, labels, chunks = build_rag_dataset(
        docs, questions, "recursive", HashingEmbedder(64), chunk_size=40, chunk_overlap=0
    )
    assert dataset.n == len(chunks) == len(labels.item_labels) >= 3
    assert dataset.n_queries == 2
    assert labels.query_targets == ["coffee.txt", "garden/compost.md"]

    [result] = run_benchmark(
        dataset, [IndexConfig("faiss", "flat", {"index": "flat"})], 2, warmup=0, relevance=labels
    )
    assert result.status == "ok"
    assert result.hit_rate_at_k == 1.0
    assert result.mrr_at_k == 1.0


def test_bundled_sample_is_consistent():
    docs = load_corpus(SAMPLE_DIR / "corpus")
    questions = load_questions(SAMPLE_DIR / "questions.jsonl", docs)
    assert len(docs) == 10
    assert len(questions) == 30
    assert {q.answer_doc for q in questions} == {d.doc_id for d in docs}
