"""
Abstract base class for multiprocessing-compatible lock implementations.

Uses multiprocessing.Value / multiprocessing.Lock for counters so that
state is shared across processes (not just threads).
"""

from __future__ import annotations

import abc
import multiprocessing
from typing import Optional


class MPBaseLock(abc.ABC):
    """Abstract base for all multiprocessing-safe lock implementations."""

    def __init__(self) -> None:
        self._lock_internal = multiprocessing.Lock()
        self._acquire_attempts  = multiprocessing.Value('l', 0)
        self._acquire_successes = multiprocessing.Value('l', 0)
        self._wait_time_ns      = multiprocessing.Value('l', 0)

    # ------------------------------------------------------------------
    # Abstract interface  (proc_idx passed by the MP worker)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def acquire(self, proc_idx: int = 0) -> None:
        """Block until the lock is acquired."""

    @abc.abstractmethod
    def release(self, proc_idx: int = 0) -> None:
        """Release the lock."""

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    def reset_counters(self) -> None:
        with self._lock_internal:
            self._acquire_attempts.value  = 0
            self._acquire_successes.value = 0
            self._wait_time_ns.value      = 0

    @property
    def acquire_attempts(self) -> int:
        return self._acquire_attempts.value

    @property
    def acquire_successes(self) -> int:
        return self._acquire_successes.value

    @property
    def wait_time_ns(self) -> int:
        return self._wait_time_ns.value

    @property
    def contention_ratio(self) -> float:
        attempts  = self._acquire_attempts.value
        successes = self._acquire_successes.value
        if attempts == 0:
            return 0.0
        return max(0.0, (attempts - successes) / attempts)
