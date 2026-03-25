# python/crawlkit/client.py
"""
CrawlKit -- The main client.

Wraps Firecrawl's async crawl API with:
  - Budget tracking (BudgetTracker)
  - Circuit breaker protection (CircuitBreaker)
  - Crash-safe checkpointing (CrawlCheckpoint)
  - BLAKE3 content fingerprinting via Rust
  - Typed async event stream

Usage
-----
    from crawlkit import CrawlKit, BudgetConfig, CheckpointConfig

    kit = CrawlKit(
        api_key="fc-...",
        budget=BudgetConfig(max_credits=500, warn_at=400),
        checkpoint=CheckpointConfig(path=".crawl_state.json"),
    )

    async for event in kit.crawl("https://docs.example.com", limit=50):
        if event.type == "page":
            print(event.fingerprint[:8], event.page["metadata"]["sourceURL"])
        elif event.type == "completed":
            print(f"Done: {event.total_pages} pages, {event.credits_used} credits")
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from .budget import BudgetConfig, BudgetExceededError, BudgetTracker
from .checkpoint import CheckpointConfig, CrawlCheckpoint
from .circuit_breaker import CircuitBreaker, CircuitOpenError
from .events import (
    BudgetWarningEvent,
    CompletedEvent,
    CrawlEvent,
    FailedEvent,
    PageEvent,
)

logger = logging.getLogger(__name__)

# Rust extension
try:
    from .crawlkit_rs import content_fingerprint
except ImportError as _exc:
    raise RuntimeError(
        f"crawlkit Rust extension not compiled. Run: maturin develop\nOriginal error: {_exc}"
    ) from _exc

# firecrawl-py
try:
    from firecrawl import FirecrawlApp
except ImportError as _exc:
    raise RuntimeError(
        f"firecrawl-py not installed: pip install firecrawl-py\nOriginal error: {_exc}"
    ) from _exc

_DEFAULT_POLL_INTERVAL = 2.0
_MAX_POLL_INTERVAL = 10.0


@dataclass
class CrawlKit:
    """
    The main crawlkit client.

    Args:
        api_key:         Firecrawl API key. Falls back to FIRECRAWL_API_KEY env var.
        budget:          Credit budget config. No enforcement if None.
        checkpoint:      Checkpoint config. Disabled if None.
        circuit_breaker: Custom CircuitBreaker. Created with defaults if None.
        poll_interval:   Initial polling interval in seconds.
    """

    api_key: str | None = None
    budget: BudgetConfig | None = None
    checkpoint: CheckpointConfig | None = None
    circuit_breaker: CircuitBreaker | None = None
    poll_interval: float = _DEFAULT_POLL_INTERVAL

    def __post_init__(self) -> None:
        key = self.api_key or os.getenv("FIRECRAWL_API_KEY")
        if not key:
            raise ValueError("Firecrawl API key required. Pass api_key= or set FIRECRAWL_API_KEY.")
        self._firecrawl = FirecrawlApp(api_key=key)

        self._budget_tracker: BudgetTracker | None = (
            BudgetTracker(self.budget) if self.budget else None
        )

        self._cb: CircuitBreaker = self.circuit_breaker or CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            success_threshold=2,
        )

    async def crawl(
        self,
        url: str,
        *,
        limit: int = 100,
        max_depth: int = 3,
        formats: list[str] | None = None,
        **kwargs: object,
    ) -> AsyncIterator[CrawlEvent]:
        """
        Async generator -- crawl `url` and yield typed CrawlEvents.

        Args:
            url:       Root URL to crawl.
            limit:     Maximum pages.
            max_depth: Maximum link depth from root.
            formats:   Firecrawl output formats. Default: ["markdown"].
        """
        formats = formats or ["markdown"]

        # Pre-flight budget check
        if self._budget_tracker:
            try:
                self._budget_tracker.check(1)
            except BudgetExceededError as exc:
                yield FailedEvent(error=str(exc))
                return

        # Checkpoint: resume or start fresh
        checkpoint: CrawlCheckpoint | None = None
        job_id: str | None = None

        if self.checkpoint and self.checkpoint.enabled:
            checkpoint = CrawlCheckpoint.load(self.checkpoint.path)
            if checkpoint:
                job_id = checkpoint.job_id
                logger.info(
                    "Resuming crawl job %s (%d pages already done)",
                    job_id,
                    len(checkpoint.processed_urls),
                )

        if job_id is None:
            try:
                self._cb.before_call()
                response = await asyncio.to_thread(
                    self._firecrawl.async_crawl_url,
                    url,
                    params={
                        "limit": limit,
                        "maxDepth": max_depth,
                        "scrapeOptions": {"formats": formats},
                        **kwargs,
                    },
                )
                self._cb.on_success()
            except CircuitOpenError as exc:
                yield FailedEvent(error=f"Circuit open: {exc}")
                return
            except Exception as exc:
                self._cb.on_failure()
                logger.error("Failed to start crawl: %s", exc)
                yield FailedEvent(error=str(exc))
                return

            job_id = response.get("id") or response.get("jobId")
            if not job_id:
                yield FailedEvent(error=f"No job_id in response: {response}")
                return

            logger.info("Started crawl job %s for %s", job_id, url)

            if self.checkpoint and self.checkpoint.enabled:
                checkpoint = CrawlCheckpoint.new(job_id=job_id, path=self.checkpoint.path)
                checkpoint.save()

        # Polling loop
        seen_urls: set[str] = set(checkpoint.processed_urls) if checkpoint else set()
        total_pages = 0
        current_poll = self.poll_interval

        while True:
            try:
                self._cb.before_call()
                status_resp = await asyncio.to_thread(self._firecrawl.check_crawl_status, job_id)
                self._cb.on_success()
                current_poll = self.poll_interval
            except CircuitOpenError as exc:
                yield FailedEvent(error=f"Circuit open during poll: {exc}", job_id=job_id)
                return
            except Exception as exc:
                self._cb.on_failure()
                logger.warning("Poll failed (will retry): %s", exc)
                await asyncio.sleep(min(current_poll * 2, _MAX_POLL_INTERVAL))
                current_poll = min(current_poll * 2, _MAX_POLL_INTERVAL)
                continue

            status: str = status_resp.get("status", "")
            data: list[dict[str, Any]] = status_resp.get("data") or []
            credits_used: int = status_resp.get("creditsUsed", 0)

            for page in data:
                page_url: str = page.get("metadata", {}).get("sourceURL", "") or page.get("url", "")

                if page_url and page_url in seen_urls:
                    continue

                content = page.get("markdown", "") or page.get("html", "")
                fp = content_fingerprint(content) if content else ""

                if self._budget_tracker:
                    warned = self._budget_tracker.consume(1)
                    if warned:
                        yield BudgetWarningEvent(
                            credits_used=self._budget_tracker.credits_used,
                            budget=self._budget_tracker.max_credits,
                        )
                    try:
                        self._budget_tracker.check(1)
                    except BudgetExceededError as exc:
                        logger.warning("Budget exceeded mid-crawl: %s", exc)
                        yield FailedEvent(error=str(exc), job_id=job_id)
                        return

                yield PageEvent(
                    page=page,
                    fingerprint=fp,
                    credits_used=credits_used,
                )
                total_pages += 1

                if page_url:
                    seen_urls.add(page_url)
                    if checkpoint:
                        checkpoint.add_processed(page_url)
                        checkpoint.credits_used = credits_used
                        checkpoint.save()

            if status == "completed":
                if checkpoint:
                    checkpoint.delete()
                yield CompletedEvent(
                    total_pages=total_pages,
                    credits_used=credits_used,
                    job_id=job_id,
                )
                return

            if status in ("failed", "cancelled"):
                error = status_resp.get("error", f"Job {status}")
                yield FailedEvent(error=error, job_id=job_id)
                return

            await asyncio.sleep(current_poll)
