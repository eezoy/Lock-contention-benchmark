"""
Abstract base class for all lock implementations.

Every custom lock must inherit BaseLock and implement acquire() / release().
The context-manager protocol (__enter__ / __exit__) is provided here so
subclasses only need to implement the two primitive operations.
"""

from __future__ import annotations

import abc
import threading
from typing import Optional


class BaseLock(abc.ABC):
    """Abstract base class for all benchmark lock implementations."""

    def __init__(self) -> None:
        # Contention counters — updated by subclasses
        self._acquire_attempts: int = 0
        self._acquire_successes: int = 0
        self._wait_time_ns: int = 0
        self._lock_internal = threading.Lock()  # guards counter updates

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def acquire(self) -> None:
        """Block until the lock is acquired."""

    @abc.abstractmethod
    def release(self) -> None:
        """Release the lock."""

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "BaseLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> bool:
        self.release()
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Counters (thread-safe reads)
    # ------------------------------------------------------------------

    def reset_counters(self) -> None:
        with self._lock_internal:
            self._acquire_attempts = 0
            self._acquire_successes = 0
            self._wait_time_ns = 0

    @property
    def acquire_attempts(self) -> int:
        return self._acquire_attempts

    @property
    def acquire_successes(self) -> int:
        return self._acquire_successes

    @property
    def wait_time_ns(self) -> int:
        return self._wait_time_ns

    @property
    def contention_ratio(self) -> float:
        """Ratio of failed attempts to total attempts (0 = no contention)."""
        if self._acquire_attempts == 0:
            return 0.0
        failed = self._acquire_attempts - self._acquire_successes
        return failed / self._acquire_attempts
