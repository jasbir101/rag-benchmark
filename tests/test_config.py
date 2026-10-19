import os

import pytest

from rag_benchmark.config import (
    DEFAULT_INDEXES,
    build_index_configs,
    config_label,
    expand_grid,
    load_dotenv,
    load_settings,
)


def test_expand_grid():
    assert expand_grid({"index": "hnsw", "ef_search": [16, 64]}) == [
        {"index": "hnsw", "ef_search": 16},
        {"index": "hnsw", "ef_search": 64},
    ]
    assert len(expand_grid({"a": [1, 2], "b": [3, 4, 5]})) == 6
    assert expand_grid({}) == [{}]


def test_config_label():
    assert config_label({"index": "hnsw", "M": 32, "ef_search": 64}) == "hnsw M=32 efS=64"
    assert config_label({}) == "default"


def test_defaults():
    settings = load_settings()
    assert settings.k == 10 and settings.n == 20_000 and settings.dim == 384
    assert settings.backends == ["faiss", "chroma", "milvus", "pinecone"]


def test_yaml_then_overrides(tmp_path):
    path = tmp_path / "bench.yaml"
    path.write_text(
        "k: 5\n"
        "backends: [faiss]\n"
        "dataset:\n  n: 1000\n  dim: 64\n"
        "rag:\n  chunk_size: 300\n"
        "indexes:\n  faiss:\n    - {index: hnsw, ef_search: [8, 16]}\n",
        encoding="utf-8",
    )
    settings = load_settings(path, {"dim": 128, "k": None, "backends": "faiss,chroma"})
    assert settings.k == 5  # None override is ignored
    assert settings.n == 1000
    assert settings.dim == 128  # CLI wins over YAML
    assert settings.chunk_size == 300
    assert settings.backends == ["faiss", "chroma"]

    configs = build_index_configs(settings)
    faiss = [c for c in configs if c.backend == "faiss"]
    assert [c.label for c in faiss] == ["hnsw efS=8", "hnsw efS=16"]
    chroma = [c for c in configs if c.backend == "chroma"]
    assert len(chroma) == len(expand_grid(DEFAULT_INDEXES["chroma"][0]))


def test_unknown_setting_is_an_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("n_vectors: 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="n_vectors"):
        load_settings(path)


def test_unknown_backend_is_an_error():
    with pytest.raises(ValueError, match="weaviate"):
        load_settings(overrides={"backends": "faiss,weaviate"})


def test_load_dotenv_does_not_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_BENCH_TEST_EXISTING", "from-shell")
    monkeypatch.delenv("RAG_BENCH_TEST_NEW", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "RAG_BENCH_TEST_EXISTING=from-file\n"
        'export RAG_BENCH_TEST_NEW="quoted value"\n'
        "not a valid line\n",
        encoding="utf-8",
    )
    try:
        assert load_dotenv(env) == ["RAG_BENCH_TEST_NEW"]
        assert os.environ["RAG_BENCH_TEST_EXISTING"] == "from-shell"
        assert os.environ["RAG_BENCH_TEST_NEW"] == "quoted value"
    finally:
        os.environ.pop("RAG_BENCH_TEST_NEW", None)


def test_load_dotenv_missing_file(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == []
