# python/crawlkit/circuit_breaker.py
"""
Circuit breaker for Firecrawl API calls.

States
------
CLOSED     -- normal operation; failures are counted
OPEN       -- requests are rejected immediately (no API call made)
HALF_OPEN  -- one test request is allowed through; success closes, failure re-opens

Transitions
-----------
CLOSED    -> OPEN       after consecutive_failures >= failure_threshold
OPEN      -> HALF_OPEN  after time.monotonic() - opened_at >= recovery_timeout
HALF_OPEN -> CLOSED     after consecutive_successes >= success_threshold
HALF_OPEN -> OPEN       on any failure while testing

Usage
-----
    cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

    try:
        cb.before_call()           # raises CircuitOpenError if OPEN
        result = await api_call()
        cb.on_success()
    except CircuitOpenError:
        raise                      # fast-fail -- don't hit the API
    except Exception:
        cb.on_failure()
        raise
"""

from __future__ import annotations

import logging
import time
from enum import StrEnum

logger = logging.getLogger(__name__)


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised by CircuitBreaker.before_call() when the circuit is OPEN."""

    def __init__(self, retry_after: float) -> None:
        self.retry_after = retry_after
        super().__init__(f"Circuit is OPEN. Retry after {retry_after:.1f}s")


class CircuitBreaker:
    """
    Configurable 3-state circuit breaker.

    Args:
        failure_threshold:  Consecutive failures before opening.
        recovery_timeout:   Seconds to wait in OPEN before testing.
        success_threshold:  Consecutive successes in HALF_OPEN to close.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be > 0")
        if success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state: CircuitState = CircuitState.CLOSED
        self._consecutive_failures: int = 0
        self._consecutive_successes: int = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        # Lazily transition OPEN -> HALF_OPEN when timeout has elapsed.
        if (
            self._state is CircuitState.OPEN
            and self._opened_at is not None
            and time.monotonic() - self._opened_at >= self.recovery_timeout
        ):
            logger.info("Circuit breaker: OPEN -> HALF_OPEN (timeout elapsed)")
            self._state = CircuitState.HALF_OPEN
            self._consecutive_successes = 0
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def retry_after(self) -> float:
        """Seconds until HALF_OPEN transition. 0 if not OPEN."""
        if self._state is not CircuitState.OPEN or self._opened_at is None:
            return 0.0
        elapsed = time.monotonic() - self._opened_at
        return max(0.0, self.recovery_timeout - elapsed)

    def before_call(self) -> None:
        """Call BEFORE making an API request. Raises CircuitOpenError if OPEN."""
        current = self.state
        if current is CircuitState.OPEN:
            raise CircuitOpenError(retry_after=self.retry_after)

    def on_success(self) -> None:
        """Call after a successful API response."""
        self._consecutive_failures = 0
        current = self.state
        if current is CircuitState.HALF_OPEN:
            self._consecutive_successes += 1
            if self._consecutive_successes >= self.success_threshold:
                logger.info(
                    "Circuit breaker: HALF_OPEN -> CLOSED (%d successes)",
                    self._consecutive_successes,
                )
                self._state = CircuitState.CLOSED
                self._consecutive_successes = 0
                self._opened_at = None
        else:
            self._consecutive_successes = 0

    def on_failure(self) -> None:
        """Call after a failed API call (network error, 5xx, timeout)."""
        self._consecutive_failures += 1
        self._consecutive_successes = 0
        current = self.state
        if current is CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker: HALF_OPEN -> OPEN (failure during probe)")
            self._transition_to_open()
        elif current is CircuitState.CLOSED:
            if self._consecutive_failures >= self.failure_threshold:
                logger.warning(
                    "Circuit breaker: CLOSED -> OPEN (%d consecutive failures)",
                    self._consecutive_failures,
                )
                self._transition_to_open()

    def _transition_to_open(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()

    def reset(self) -> None:
        """Force reset to CLOSED -- useful for testing."""
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._opened_at = None

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(state={self.state.value}, "
            f"failures={self._consecutive_failures}/{self.failure_threshold})"
        )
