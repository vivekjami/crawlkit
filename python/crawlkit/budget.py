# python/crawlkit/budget.py
"""
Credit-budget tracking for Firecrawl API calls.

Usage:
    config  = BudgetConfig(max_credits=500, warn_at=400)
    tracker = BudgetTracker(config)

    tracker.check(1)     # raises BudgetExceededError if over limit
    tracker.consume(1)   # records spend; fires warning callback at threshold
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when a crawl operation would exceed the configured credit limit."""

    def __init__(self, used: int, requested: int, limit: int) -> None:
        self.used = used
        self.requested = requested
        self.limit = limit
        super().__init__(f"Budget exceeded: {used} used + {requested} requested > {limit} limit")


@dataclass
class BudgetConfig:
    """
    Immutable budget settings.

    Args:
        max_credits:  Hard ceiling. CrawlKit will not start a crawl that would
                      push usage past this value.
        warn_at:      Emit a BudgetWarningEvent when credits_used reaches this
                      threshold. Defaults to 80% of max_credits.
    """

    max_credits: int
    warn_at: int | None = None

    def __post_init__(self) -> None:
        if self.max_credits <= 0:
            raise ValueError("max_credits must be > 0")
        if self.warn_at is None:
            self.warn_at = int(self.max_credits * 0.8)
        if self.warn_at > self.max_credits:
            raise ValueError("warn_at cannot exceed max_credits")


class BudgetTracker:
    """
    Mutable runtime state for one crawl session.

    Thread-safety note: CrawlKit is single-threaded async; no locks needed.
    """

    def __init__(
        self,
        config: BudgetConfig,
        on_warning: Callable[[int, int], None] | None = None,
    ) -> None:
        self._config = config
        self._credits_used: int = 0
        self._warning_fired: bool = False
        self._on_warning = on_warning

    @property
    def credits_used(self) -> int:
        return self._credits_used

    @property
    def max_credits(self) -> int:
        return self._config.max_credits

    @property
    def warn_at(self) -> int:
        return self._config.warn_at  # type: ignore[return-value]

    @property
    def remaining(self) -> int:
        return max(0, self._config.max_credits - self._credits_used)

    def check(self, requested: int = 1) -> None:
        """
        Pre-flight guard. Raises BudgetExceededError *before* making an API
        call if the spend would push usage over the limit.
        """
        if self._credits_used + requested > self._config.max_credits:
            raise BudgetExceededError(self._credits_used, requested, self._config.max_credits)

    def consume(self, amount: int = 1) -> bool:
        """
        Record actual spend. Returns True if the warning threshold was just
        crossed for the first time (caller should yield a BudgetWarningEvent).
        """
        if amount < 0:
            raise ValueError("amount must be >= 0")
        self._credits_used += amount
        logger.debug(
            "Budget: %d / %d credits used",
            self._credits_used,
            self._config.max_credits,
        )
        crossed = (
            not self._warning_fired and self._credits_used >= self._config.warn_at  # type: ignore[operator]
        )
        if crossed:
            self._warning_fired = True
            if self._on_warning:
                self._on_warning(self._credits_used, self._config.max_credits)
            logger.warning(
                "Budget warning: %d / %d credits used (%.0f%%)",
                self._credits_used,
                self._config.max_credits,
                100 * self._credits_used / self._config.max_credits,
            )
        return crossed

    def reset(self) -> None:
        """Reset state -- useful for resuming a saved session."""
        self._credits_used = 0
        self._warning_fired = False
