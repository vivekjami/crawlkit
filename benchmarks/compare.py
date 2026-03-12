"""
crawlkit benchmark

Benchmarks crawlkit-rs vs pure Python implementations
on a real dataset.

Run:
    pip install datasets blake3
    python benchmarks/compare.py
"""

import random
import re
import time
from statistics import median
from urllib.parse import urlparse

from blake3 import blake3
from crawlkit import (
    batch_chunk_documents,
    batch_fingerprint_pages,
    normalize_and_dedup,
)
from datasets import load_dataset

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

N_PAGES = 100_000
RUNS = 5
RANDOM_SEED = 42


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────


def load_public_dataset():
    print(f"\nLoading dataset: CC News ({N_PAGES:,} pages)")

    ds = load_dataset("cc_news", split=f"train[:{N_PAGES}]")

    pages = [(row["url"], row["text"]) for row in ds]
    urls = [row["url"] for row in ds]

    random.seed(RANDOM_SEED)

    varied_urls = []

    for url in urls:
        v = url

        if random.random() < 0.3:
            v += "#fragment"

        if random.random() < 0.2:
            v += "?utm_source=test&utm_medium=benchmark"

        varied_urls.append(v)

    size_mb = sum(len(t) for _, t in pages) / 1e6

    print(f"✓ Loaded {len(pages):,} pages")
    print(f"✓ Dataset size: {size_mb:.1f} MB\n")

    return pages, varied_urls


# ─────────────────────────────────────────────
# Python baseline implementations
# ─────────────────────────────────────────────

DATE_RE = re.compile(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b")
TS_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def py_fingerprint(content):
    s = DATE_RE.sub("__DATE__", content)
    s = TS_RE.sub("__TIMESTAMP__", s)
    s = " ".join(s.split())

    return blake3(s.encode()).hexdigest()


def py_batch_fingerprint(pages):
    return [(u, py_fingerprint(t)) for u, t in pages]


def py_chunk_text(text, max_chars=1500):
    """Mirror of chunker.rs chunk_markdown logic."""
    chunks = []
    current_section = "root"
    current_content = []
    chunk_index = 0

    def flush_section(section, content_lines):
        nonlocal chunk_index
        trimmed = "\n".join(content_lines).strip()
        if not trimmed:
            return
        if len(trimmed) <= max_chars:
            chunks.append(
                {
                    "content": trimmed,
                    "section": section,
                    "char_count": len(trimmed),
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1
        else:
            # Oversized: split at paragraph boundaries
            for sub in split_by_paragraphs(trimmed, max_chars):
                chunks.append(
                    {
                        "content": sub,
                        "section": section,
                        "char_count": len(sub),
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1

    def split_by_paragraphs(content, max_chars):
        result = []
        current = ""
        for paragraph in content.split("\n\n"):
            para = paragraph.strip()
            if not para:
                continue
            if current and len(current) + len(para) + 2 > max_chars:
                result.append(current.strip())
                current = ""
            if current:
                current += "\n\n"
            current += para
        if current.strip():
            result.append(current.strip())
        if not result:
            result.append(content.strip())
        return result

    for line in text.splitlines():
        if line.startswith("#"):
            flush_section(current_section, current_content)
            current_content = []
            current_section = line.lstrip("#").strip()
        else:
            current_content.append(line)

    flush_section(current_section, current_content)
    return chunks


def py_batch_chunk(pages):
    return [(u, py_chunk_text(t)) for u, t in pages]


def py_normalize_and_dedup(urls):
    seen = set()
    out = []

    for u in urls:
        try:
            p = urlparse(u)

            if p.scheme not in ("http", "https"):
                continue

            key = f"{p.scheme}://{p.netloc}{p.path}"

            if key not in seen:
                seen.add(key)
                out.append(key)

        except Exception:
            pass

    return out


# ─────────────────────────────────────────────
# Benchmark helpers
# ─────────────────────────────────────────────


def bench(fn, *args):
    times = []

    for _ in range(RUNS):
        t0 = time.perf_counter()

        fn(*args)

        times.append((time.perf_counter() - t0) * 1000)

    return median(times)


COLS = {
    "op": 28,
    "py": 14,
    "rs": 14,
    "speedup": 9,
    "py_rate": 13,
    "rs_rate": 13,
}

SEP = "─" * sum(COLS.values())


def print_row(name, py, rs):
    speedup = py / rs
    py_rate = int(N_PAGES / (py / 1000))
    rs_rate = int(N_PAGES / (rs / 1000))

    py_str = f"{py:,.1f} ms"
    rs_str = f"{rs:,.1f} ms"
    sp_str = f"{speedup:.1f}x"
    pyr_str = f"{py_rate:,}"
    rsr_str = f"{rs_rate:,}"

    print(
        f"{name:<{COLS['op']}}"
        f"{py_str:>{COLS['py']}}"
        f"{rs_str:>{COLS['rs']}}"
        f"{sp_str:>{COLS['speedup']}}"
        f"{pyr_str:>{COLS['py_rate']}}"
        f"{rsr_str:>{COLS['rs_rate']}}"
    )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────


def main():
    pages, urls = load_public_dataset()

    print("Verifying outputs...\n")

    # fingerprint correctness
    py_fp = py_batch_fingerprint(pages)
    rs_fp = batch_fingerprint_pages(pages)

    assert len(py_fp) == len(rs_fp)

    print("✓ Fingerprint count matches")

    # dedup sanity check
    py_ud = py_normalize_and_dedup(urls)
    rs_ud = normalize_and_dedup(urls)

    print(f"✓ Python dedup count: {len(py_ud):,}")
    print(f"✓ Rust dedup count:   {len(rs_ud):,}\n")

    print("Running warmup...")

    batch_fingerprint_pages(pages)
    batch_chunk_documents(pages, 1500)
    normalize_and_dedup(urls)

    print("\nBenchmarking...\n")

    print(SEP)

    print(
        f"{'Operation':<{COLS['op']}}"
        f"{'Python':>{COLS['py']}}"
        f"{'crawlkit-rs':>{COLS['rs']}}"
        f"{'Speedup':>{COLS['speedup']}}"
        f"{'Py pages/s':>{COLS['py_rate']}}"
        f"{'Rs pages/s':>{COLS['rs_rate']}}"
    )

    print(SEP)

    py_fp_t = bench(py_batch_fingerprint, pages)
    rs_fp_t = bench(batch_fingerprint_pages, pages)

    print_row("Content fingerprint", py_fp_t, rs_fp_t)

    py_ck_t = bench(py_batch_chunk, pages)
    rs_ck_t = bench(batch_chunk_documents, pages, 1500)

    print_row("Document chunking", py_ck_t, rs_ck_t)

    py_ud_t = bench(py_normalize_and_dedup, urls)
    rs_ud_t = bench(normalize_and_dedup, urls)

    print_row("URL normalize + dedup", py_ud_t, rs_ud_t)

    print(SEP)

    print(f"\nMedian of {RUNS} runs")
    print("Python = serial execution (GIL)")
    print("Rust = parallel execution via Rayon\n")


if __name__ == "__main__":
    main()
