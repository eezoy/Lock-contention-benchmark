"""
Multiprocessing-compatible Readers-Writer Lock (writer-preference).

All shared state uses multiprocessing.Value (shared memory) and
multiprocessing.Condition (semaphore-backed) so that multiple OS
processes can synchronise correctly.

In this benchmark every worker performs a read-modify-write (increment),
so acquire() always takes the *write* path.  The lock degenerates to a
mutex for this workload — which is the expected demonstration that RWLock
has higher overhead than a plain mutex on write-heavy workloads.
"""

from __future__ import annotations

import time
import multiprocessing

from benchmark.locks.mp_base import MPBaseLock


class MPRWLock(MPBaseLock):
    """Writer-preference RW lock for multiprocessing."""

    def __init__(self) -> None:
        super().__init__()
        # Condition variable: _cond._lock is the mutex that guards
        # all state variables below.
        self._cond          = multiprocessing.Condition()
        # All three counters use lock=False because they are only ever
        # accessed while _cond is held, which serialises all access.
        self._readers       = multiprocessing.Value('i', 0, lock=False)
        self._writer_active = multiprocessing.Value('i', 0, lock=False)
        self._write_pending = multiprocessing.Value('i', 0, lock=False)

    def acquire(self, proc_idx: int = 0) -> None:
        """Acquire in write (exclusive) mode."""
        with self._lock_internal:
            self._acquire_attempts.value += 1

        t_start = time.perf_counter_ns()

        with self._cond:
            self._write_pending.value += 1
            while self._readers.value > 0 or self._writer_active.value:
                self._cond.wait()
            self._write_pending.value -= 1
            self._writer_active.value  = 1

        elapsed = time.perf_counter_ns() - t_start
        with self._lock_internal:
            self._acquire_successes.value += 1
            self._wait_time_ns.value      += elapsed

    def release(self, proc_idx: int = 0) -> None:
        with self._cond:
            self._writer_active.value = 0
            self._cond.notify_all()

    # ------------------------------------------------------------------
    # Read path (not used by the benchmark harness, provided for
    # completeness so the class can be used as a real RW lock)
    # ------------------------------------------------------------------

    def acquire_read(self, proc_idx: int = 0) -> None:
        with self._cond:
            while self._writer_active.value or self._write_pending.value > 0:
                self._cond.wait()
            self._readers.value += 1

    def release_read(self, proc_idx: int = 0) -> None:
        with self._cond:
            self._readers.value -= 1
            if self._readers.value == 0:
                self._cond.notify_all()
