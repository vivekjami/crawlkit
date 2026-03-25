# tests/test_circuit_breaker.py
import time

import pytest
from crawlkit.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


class TestCircuitBreakerInit:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state is CircuitState.CLOSED

    def test_invalid_failure_threshold(self):
        with pytest.raises(ValueError):
            CircuitBreaker(failure_threshold=0)

    def test_invalid_recovery_timeout(self):
        with pytest.raises(ValueError):
            CircuitBreaker(recovery_timeout=0)


class TestCircuitOpensAfterFailures:
    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.on_failure()
        assert cb.state is CircuitState.OPEN

    def test_does_not_open_before_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(2):
            cb.on_failure()
        assert cb.state is CircuitState.CLOSED

    def test_failure_count_resets_on_success(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.on_failure()
        cb.on_failure()
        cb.on_success()
        cb.on_failure()
        cb.on_failure()
        assert cb.state is CircuitState.CLOSED


class TestCircuitRejectsWhileOpen:
    def test_before_call_raises_when_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        cb.on_failure()
        assert cb.state is CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            cb.before_call()

    def test_circuit_open_error_has_retry_after(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=30.0)
        cb.on_failure()
        with pytest.raises(CircuitOpenError) as exc_info:
            cb.before_call()
        assert exc_info.value.retry_after > 0

    def test_before_call_passes_when_closed(self):
        cb = CircuitBreaker()
        cb.before_call()


class TestHalfOpenTransition:
    def test_transitions_to_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.001)
        cb.on_failure()
        assert cb._state is CircuitState.OPEN
        time.sleep(0.01)
        assert cb.state is CircuitState.HALF_OPEN

    def test_stays_open_before_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=9999.0)
        cb.on_failure()
        assert cb.state is CircuitState.OPEN

    def test_half_open_allows_before_call(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.001)
        cb.on_failure()
        time.sleep(0.01)
        cb.before_call()


class TestCircuitClosesFromHalfOpen:
    def _half_open(self, success_threshold: int = 2) -> CircuitBreaker:
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.001,
            success_threshold=success_threshold,
        )
        cb.on_failure()
        time.sleep(0.01)
        _ = cb.state
        return cb

    def test_closes_after_successes(self):
        cb = self._half_open(success_threshold=2)
        assert cb.state is CircuitState.HALF_OPEN
        cb.on_success()
        cb.on_success()
        assert cb.state is CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        cb = self._half_open(success_threshold=2)
        cb.on_success()
        cb.on_failure()
        assert cb.state is CircuitState.OPEN

    def test_does_not_close_before_threshold(self):
        cb = self._half_open(success_threshold=3)
        cb.on_success()
        cb.on_success()
        assert cb.state is CircuitState.HALF_OPEN


class TestCircuitBreakerReset:
    def test_reset_returns_to_closed(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.on_failure()
        cb.reset()
        assert cb.state is CircuitState.CLOSED
        assert cb.consecutive_failures == 0
