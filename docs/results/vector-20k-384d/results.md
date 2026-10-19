# rag-benchmark results

- Mode: `vector`
- Dataset: synthetic n=20000 dim=384 clusters=64 std=0.05 seed=42
- Queries: 1000, k = 10, metric: cosine
- CPU: 13th Gen Intel(R) Core(TM) i7-1355U (12 logical cores)
- RAM: 15.7 GB
- OS: Windows-11-10.0.26300-SP0, Python 3.13.14
- Started: 2026-10-19T15:41:36

| Backend | Config | Build (s) | Recall@10 | p50 (ms) | p95 (ms) | p99 (ms) | QPS (1 client) | QPS (batch) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| faiss | flat | 0.03 | 1.0000 | 1.976 | 2.554 | 4.227 | 501 | 8,158 |
| faiss | hnsw M=32 efC=200 efS=16 | 3.79 | 0.9135 | 0.250 | 0.347 | 0.418 | 3,904 | 29,736 |
| faiss | hnsw M=32 efC=200 efS=32 | 3.48 | 0.9842 | 0.337 | 0.450 | 0.600 | 2,904 | 19,920 |
| faiss | hnsw M=32 efC=200 efS=64 | 3.68 | 0.9992 | 0.545 | 0.831 | 0.948 | 1,740 | 8,164 |
| faiss | hnsw M=32 efC=200 efS=128 | 3.79 | 1.0000 | 0.842 | 1.156 | 1.391 | 1,153 | 4,760 |
| faiss | hnsw M=32 efC=200 efS=256 | 3.58 | 1.0000 | 1.515 | 2.175 | 2.840 | 633 | 4,244 |
| faiss | ivf_flat nlist=256 nprobe=2 | 0.53 | 0.7782 | 0.169 | 0.285 | 0.390 | 5,433 | 45,496 |
| faiss | ivf_flat nlist=256 nprobe=4 | 0.54 | 0.9581 | 0.185 | 0.292 | 0.379 | 5,130 | 25,333 |
| faiss | ivf_flat nlist=256 nprobe=8 | 0.60 | 0.9987 | 0.290 | 0.471 | 0.700 | 3,239 | 10,513 |
| faiss | ivf_flat nlist=256 nprobe=16 | 0.69 | 1.0000 | 0.482 | 0.805 | 0.942 | 1,967 | 5,683 |
| faiss | ivf_flat nlist=256 nprobe=32 | 0.86 | 1.0000 | 1.030 | 1.433 | 1.643 | 957 | 5,569 |
| faiss | ivf_flat nlist=256 nprobe=64 | 0.69 | 1.0000 | 1.886 | 2.892 | 3.908 | 501 | 2,554 |
| chroma | M=16 efC=100 efS=16 | 4.38 | 0.8820 | 1.302 | 1.636 | 1.826 | 799 | 3,469 |
| chroma | M=16 efC=100 efS=32 | 4.25 | 0.9751 | 1.218 | 1.796 | 2.243 | 803 | 3,322 |
| chroma | M=16 efC=100 efS=64 | 3.95 | 0.9975 | 1.483 | 2.063 | 2.654 | 645 | 2,653 |
| chroma | M=16 efC=100 efS=128 | 5.09 | 0.9999 | 1.462 | 1.866 | 2.697 | 715 | 2,295 |
| chroma | M=16 efC=100 efS=256 | 3.83 | 1.0000 | 1.551 | 2.071 | 2.303 | 652 | 1,937 |

Not run:

- **milvus** skipped: MILVUS_URI is not set (Milvus server or Zilliz Cloud URI)
- **pinecone** skipped: PINECONE_API_KEY is not set

![recall vs p95 latency](recall_vs_p95.png)
