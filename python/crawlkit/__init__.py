# python/crawlkit/__init__.py
"""
crawlkit -- Production reliability layer for async web crawling pipelines.

Rust-accelerated core operations via PyO3 + Rayon.
Python async orchestration layer with budget tracking, circuit breaking,
crash-safe checkpointing, and a typed event stream.
"""

# Python orchestration layer
from .budget import BudgetConfig, BudgetExceededError, BudgetTracker
from .checkpoint import CheckpointConfig, CrawlCheckpoint
from .circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from .client import CrawlKit
from .events import (
    BudgetWarningEvent,
    CompletedEvent,
    CrawlEvent,
    FailedEvent,
    PageEvent,
)
from .webhook import WebhookConfig, WebhookServer

# Rust extension -- fail loudly if not compiled
try:
    from .crawlkit_rs import (
        batch_chunk_documents,
        batch_fingerprint_pages,
        chunk_markdown,
        content_fingerprint,
        content_has_changed,
        extract_clean_text,
        filter_uncrawled,
        normalize_and_dedup,
        normalize_url,
        token_comparison,
    )
except ImportError as _e:
    raise RuntimeError(
        f"crawlkit Rust extension not compiled.\nRun:  maturin develop\nOriginal error: {_e}"
    ) from _e

__version__ = "0.2.0"

__all__ = [
    # Main client
    "CrawlKit",
    # Config
    "BudgetConfig",
    "BudgetTracker",
    "BudgetExceededError",
    "CheckpointConfig",
    "CrawlCheckpoint",
    "CircuitBreaker",
    "CircuitState",
    "CircuitOpenError",
    # Webhook
    "WebhookServer",
    "WebhookConfig",
    # Events
    "CrawlEvent",
    "PageEvent",
    "CompletedEvent",
    "FailedEvent",
    "BudgetWarningEvent",
    # Rust core
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
]
