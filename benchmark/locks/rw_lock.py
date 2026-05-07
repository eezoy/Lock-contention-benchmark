"""
RW-Lock — readers/writer lock.

Algorithm
---------
Multiple threads may hold the *read* lock concurrently, but a writer
requires exclusive access (no readers, no other writer).

State:
  - _readers: number of active readers (int, protected by _state_lock)
  - _writer_active: bool — a writer currently holds the lock
  - _write_pending: number of writers waiting (to prevent reader starvation)

Acquire read:
  Wait until no writer is active AND no writer is pending, then increment
  _readers.

Acquire write:
  Increment _write_pending, wait until no readers and no writer, then
  take exclusive access and decrement _write_pending.

Release:
  Decrement the appropriate counter and notify_all().

Properties
----------
- High throughput for read-heavy workloads.
- Writers are not starved: pending writers block new readers.
- Not FIFO between multiple writers (condition variable wakeup order is
  implementation-defined).

Benchmark note
--------------
In this benchmark every thread does a read-modify-write (increment), so
we always acquire the *write* lock.  The RWLock therefore degenerates to
a mutex for this workload — which is expected and illustrates the cost of
the more complex state machine compared to a plain threading.Lock.
"""

from __future__ import annotations

import time
import threading

from benchmark.locks.base import BaseLock


class RWLock(BaseLock):
    """Readers-writer lock with writer-preference."""

    def __init__(self) -> None:
        super().__init__()
        self._state_lock = threading.Lock()
        self._cond = threading.Condition(self._state_lock)
        self._readers: int = 0
        self._writer_active: bool = False
        self._write_pending: int = 0

    # ------------------------------------------------------------------
    # Public API — readers
    # ------------------------------------------------------------------

    def acquire_read(self) -> None:
        """Acquire the lock in read (shared) mode."""
        with self._lock_internal:
            self._acquire_attempts += 1

        t_start = time.perf_counter_ns()
        with self._cond:
            while self._writer_active or self._write_pending > 0:
                self._cond.wait()
            self._readers += 1

        elapsed = time.perf_counter_ns() - t_start
        with self._lock_internal:
            self._acquire_successes += 1
            self._wait_time_ns += elapsed

    def release_read(self) -> None:
        """Release a previously acquired read lock."""
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    # ------------------------------------------------------------------
    # Public API — writers (used by BaseLock acquire/release)
    # ------------------------------------------------------------------

    def acquire_write(self) -> None:
        """Acquire the lock in write (exclusive) mode."""
        with self._lock_internal:
            self._acquire_attempts += 1

        t_start = time.perf_counter_ns()
        with self._cond:
            self._write_pending += 1
            while self._readers > 0 or self._writer_active:
                self._cond.wait()
            self._write_pending -= 1
            self._writer_active = True

        elapsed = time.perf_counter_ns() - t_start
        with self._lock_internal:
            self._acquire_successes += 1
            self._wait_time_ns += elapsed

    def release_write(self) -> None:
        """Release the write lock."""
        with self._cond:
            self._writer_active = False
            self._cond.notify_all()

    # ------------------------------------------------------------------
    # BaseLock interface — delegates to write mode for benchmark harness
    # ------------------------------------------------------------------

    def acquire(self) -> None:
        self.acquire_write()

    def release(self) -> None:
        self.release_write()

    def __repr__(self) -> str:
        return "RWLock()"
