"""
Multiprocessing-compatible Spinlock.

With real OS-level parallelism (no GIL) the busy-wait phase actually
burns CPU cycles on multiple cores simultaneously — exactly the behaviour
a spinlock exhibits in C/C++/Rust.
"""

from __future__ import annotations

import time
import multiprocessing

from benchmark.locks.mp_base import MPBaseLock

# More spin iterations than the threading version: without GIL starvation
# a longer spin phase lowers latency under low contention.
_SPIN_COUNT = 64


class MPSpinlock(MPBaseLock):
    """Busy-wait spinlock backed by a multiprocessing.Lock."""

    def __init__(self) -> None:
        super().__init__()
        self._flag = multiprocessing.Lock()

    def acquire(self, proc_idx: int = 0) -> None:
        with self._lock_internal:
            self._acquire_attempts.value += 1

        t_start = time.perf_counter_ns()

        for _ in range(_SPIN_COUNT):
            if self._flag.acquire(block=False):
                elapsed = time.perf_counter_ns() - t_start
                with self._lock_internal:
                    self._acquire_successes.value += 1
                    self._wait_time_ns.value      += elapsed
                return

        # Fall back to OS-assisted blocking after spin budget exhausted
        self._flag.acquire()
        elapsed = time.perf_counter_ns() - t_start
        with self._lock_internal:
            self._acquire_successes.value += 1
            self._wait_time_ns.value      += elapsed

    def release(self, proc_idx: int = 0) -> None:
        self._flag.release()
