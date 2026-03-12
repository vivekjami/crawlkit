# crawlkit
Production reliability layer for async web crawling pipelines.
Rust-accelerated core for fingerprinting, chunking, URL deduplication, and HTML extraction.

## Performance
> Benchmark: 100,000 CC News pages | AMD Ryzen 7 3700U (4c/8t) | `maturin develop --release`

| Operation                          | Pure Python  | crawlkit   | Speedup |
|------------------------------------|--------------|------------|---------|
| Content fingerprint (100k pages)   | 15,828.6 ms  | 884.8 ms   | 17.9x   |
| Document chunking (100k pages)     | 1,570.1 ms   | 987.2 ms   | 1.6x    |
| URL normalize + dedup (100k URLs)  | 842.0 ms     | 172.4 ms   | 4.9x    |

Rust core via [PyO3](https://pyo3.rs) + [Rayon](https://docs.rs/rayon). GIL released during parallel operations.

## Installation
Requires Rust and [maturin](https://github.com/PyO3/maturin).
```bash
git clone https://github.com/vivekjami/crawlkit
cd crawlkit
pip install maturin
maturin develop --release
```

## Quick start
```python
from crawlkit import batch_fingerprint_pages, batch_chunk_documents, normalize_and_dedup

pages = [("https://example.com", "page content...")]

fingerprints = batch_fingerprint_pages(pages)
chunks       = batch_chunk_documents(pages, max_chars=1500)
urls         = normalize_and_dedup(["https://example.com?utm_source=x"])
```

## Benchmark it yourself
```bash
pip install datasets blake3
python benchmarks/compare.py
```