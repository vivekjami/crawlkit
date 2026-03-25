# python/crawlkit/events.py
"""
Typed event dataclasses yielded by CrawlKit.crawl().

Every event has a `type` field so callers can branch with a simple
if/elif or match statement without isinstance() checks.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CrawlEvent:
    """Base class. All events share `type` and an optional `message`."""

    type: str
    message: str = ""


@dataclass
class PageEvent(CrawlEvent):
    """
    Emitted once per successfully crawled page.

    `page` is the raw Firecrawl page dict (has 'markdown', 'html', 'metadata').
    `fingerprint` is the BLAKE3 hex hash of the normalised content.
    `credits_used` is the running total at the moment this page was received.
    """

    type: str = "page"
    page: dict[str, Any] = field(default_factory=dict)
    credits_used: int = 0
    fingerprint: str = ""


@dataclass
class CompletedEvent(CrawlEvent):
    """
    Emitted once when the crawl finishes successfully.

    `total_pages` is the count of pages yielded as PageEvents.
    `credits_used` is the final credit total for the job.
    `job_id` is the Firecrawl job identifier (useful for auditing).
    """

    type: str = "completed"
    total_pages: int = 0
    credits_used: int = 0
    job_id: str = ""


@dataclass
class FailedEvent(CrawlEvent):
    """
    Emitted when the crawl job itself fails (not a single-page error).

    `error` is the human-readable error string from Firecrawl.
    `job_id` identifies which job failed.
    """

    type: str = "failed"
    error: str = ""
    job_id: str = ""


@dataclass
class BudgetWarningEvent(CrawlEvent):
    """
    Emitted when credit consumption crosses the `warn_at` threshold.

    Callers can log/alert without stopping the crawl.
    """

    type: str = "budget_warning"
    credits_used: int = 0
    budget: int = 0
