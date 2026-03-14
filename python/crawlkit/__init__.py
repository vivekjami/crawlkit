"""
crawlkit — Production reliability layer for async web crawling pipelines.

Rust-accelerated core operations via PyO3 + Rayon.
"""

from crawlkit.crawlkit_rs import (
    batch_chunk_documents,
    batch_extract_clean_text,
    batch_fingerprint_pages,
    batch_token_comparison_pages,
    # Chunking
    chunk_markdown,
    # Fingerprinting
    content_fingerprint,
    content_has_changed,
    # HTML extraction
    extract_clean_text,
    filter_uncrawled,
    normalize_and_dedup,
    # URL utilities
    normalize_url,
    token_comparison,
)

__all__ = [
    "content_fingerprint",
    "content_has_changed",
    "batch_fingerprint_pages",
    "chunk_markdown",
    "batch_chunk_documents",
    "normalize_url",
    "normalize_and_dedup",
    "filter_uncrawled",
    "extract_clean_text",
    "token_comparison",
    "batch_extract_clean_text",
    "batch_token_comparison_pages",
]

__version__ = "0.1.0"
