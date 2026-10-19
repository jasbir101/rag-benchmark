from __future__ import annotations

from pathlib import Path

import pytest

from rag_benchmark.datasets import VectorDataset, make_synthetic

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample"

_SERVICE_ENV = (
    "GEMINI_API_KEY",
    "GEMINI_EMBED_MODEL",
    "GEMINI_EMBED_DIM",
    "MILVUS_URI",
    "MILVUS_TOKEN",
    "PINECONE_API_KEY",
    "PINECONE_INDEX",
    "PINECONE_CLOUD",
    "PINECONE_REGION",
)


@pytest.fixture(autouse=True)
def _no_service_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never see real credentials, so nothing can reach a network service."""
    for name in _SERVICE_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def small_dataset() -> VectorDataset:
    return make_synthetic(n=2_000, dim=32, n_queries=50, n_clusters=16, seed=7)


@pytest.fixture(scope="session")
def small_l2_dataset() -> VectorDataset:
    return make_synthetic(n=1_500, dim=24, n_queries=40, n_clusters=12, seed=11, metric="l2")
