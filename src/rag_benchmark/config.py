"""Settings: built-in defaults, overridden by a YAML file, overridden by CLI flags."""

from __future__ import annotations

import itertools
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from rag_benchmark.adapters import BACKENDS, IndexConfig
from rag_benchmark.datasets import Metric

# Index sweeps used when the YAML file does not define `indexes:` for a backend.
# A list value is a sweep: every combination becomes its own config.
DEFAULT_INDEXES: dict[str, list[dict[str, Any]]] = {
    "faiss": [
        {"index": "flat"},
        {"index": "hnsw", "M": 32, "ef_construction": 200, "ef_search": [16, 64, 256]},
        # "auto" scales nlist with the dataset (4*sqrt(n)), so small corpora work too.
        {"index": "ivf_flat", "nlist": "auto", "nprobe": [4, 16, 64]},
    ],
    "chroma": [
        {"M": 16, "ef_construction": 100, "ef_search": [16, 64, 256]},
    ],
    "milvus": [
        {"index": "hnsw", "M": 32, "ef_construction": 200, "ef_search": [16, 64, 256]},
        {"index": "ivf_flat", "nlist": 256, "nprobe": [4, 16, 64]},
    ],
    "pinecone": [{}],
}

PARAM_ABBREVIATIONS = {"ef_construction": "efC", "ef_search": "efS"}


@dataclass
class Settings:
    # common
    k: int = 10
    metric: Metric = "cosine"
    warmup: int = 20
    backends: list[str] = field(default_factory=lambda: list(BACKENDS))
    indexes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    output_dir: str = "results"
    chart: bool = True
    # vector mode
    n: int = 20_000
    dim: int = 384
    clusters: int = 64
    cluster_std: float = 0.05
    queries: int = 1_000
    seed: int = 42
    npy: str | None = None
    queries_npy: str | None = None
    # rag mode
    corpus: str | None = None
    questions: str | None = None
    embedder: str = "hash"
    embed_model: str | None = None
    embed_dim: int | None = None
    chunkers: list[str] = field(default_factory=lambda: ["fixed", "recursive"])
    chunk_size: int = 800
    chunk_overlap: int = 100
    cache_dir: str | None = ".cache/embeddings"

    def validate(self) -> None:
        unknown = [b for b in self.backends if b not in BACKENDS]
        if unknown:
            raise ValueError(f"unknown backend(s) {unknown}; choose from {sorted(BACKENDS)}")
        if self.metric not in ("cosine", "l2"):
            raise ValueError(f"metric must be cosine or l2, got {self.metric!r}")
        if self.k <= 0 or self.warmup < 0:
            raise ValueError("k must be positive and warmup non-negative")


_SECTIONS = ("dataset", "rag")


def _flatten(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Allow `dataset:` and `rag:` sections in YAML; they map onto flat settings."""
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _SECTIONS:
            if not isinstance(value, Mapping):
                raise ValueError(f"`{key}` must be a mapping")
            flat.update(value)
        else:
            flat[key] = value
    return flat


def _split_list(value: Any) -> Any:
    """Accept `a,b` strings for list settings (CLI) as well as YAML lists."""
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


def load_settings(
    yaml_path: str | Path | None = None, overrides: Mapping[str, Any] | None = None
) -> Settings:
    """Defaults < YAML file < ``overrides`` (``None`` values in overrides are ignored)."""
    merged: dict[str, Any] = {}
    if yaml_path is not None:
        with Path(yaml_path).open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"{yaml_path}: top level must be a mapping")
        merged.update(_flatten(raw))
    merged.update({k: v for k, v in (overrides or {}).items() if v is not None})

    known = {f.name for f in fields(Settings)}
    unknown = sorted(set(merged) - known)
    if unknown:
        raise ValueError(f"unknown setting(s): {', '.join(unknown)}")
    for key in ("backends", "chunkers"):
        if key in merged:
            merged[key] = _split_list(merged[key])

    settings = Settings(**merged)
    settings.validate()
    return settings


def expand_grid(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``{"a": 1, "b": [2, 3]}`` -> ``[{"a": 1, "b": 2}, {"a": 1, "b": 3}]``."""
    keys = list(params)
    choices = [v if isinstance(v, list) else [v] for v in params.values()]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*choices)]


def config_label(params: Mapping[str, Any]) -> str:
    """Short human label, e.g. ``hnsw M=32 efC=200 efS=64``."""
    parts = [str(params["index"])] if "index" in params else []
    parts += [f"{PARAM_ABBREVIATIONS.get(k, k)}={v}" for k, v in params.items() if k != "index"]
    return " ".join(parts) or "default"


def build_index_configs(settings: Settings) -> list[IndexConfig]:
    """Expand the index sweeps of every selected backend into concrete configs."""
    configs: list[IndexConfig] = []
    for backend in settings.backends:
        entries = settings.indexes.get(backend, DEFAULT_INDEXES[backend])
        for entry in entries:
            entry = dict(entry or {})
            for params in expand_grid(entry):
                configs.append(IndexConfig(backend, config_label(params), params))
    return configs


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Minimal ``.env`` reader: KEY=VALUE lines, never overrides existing variables.

    Returns the names that were set.
    """
    env_file = Path(path)
    if not env_file.is_file():
        return []
    loaded = []
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
