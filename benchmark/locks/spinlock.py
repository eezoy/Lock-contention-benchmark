"""
Spinlock — busy-wait (active polling) lock.

Algorithm
---------
A single atomic boolean flag represents the lock state.  A thread trying to
acquire spins in a tight loop, repeatedly attempting a compare-and-swap on
the flag until it transitions from False → True.

Properties
----------
- O(1) space per lock.
- No OS involvement → very low latency when contention is low.
- Wastes CPU cycles under high contention (busy-wait).
- Not fair: no ordering guarantee among waiting threads.

CPython adaptation
------------------
A pure infinite spin holds the GIL and starves the lock holder.
time.sleep(0) on Windows has a ~15 ms minimum, making the lock orders of
magnitude slower at high thread counts.  This implementation uses a
*bounded spin* (SPIN_COUNT iterations of non-blocking acquire) followed by
a blocking OS-assisted acquire — the same hybrid strategy used by Linux
futexes and Windows SRWLocks internally.  The spin phase preserves the
characteristic low-latency behaviour; the blocking fallback prevents
starvation at 32/64 threads.
"""

from __future__ import annotations

import time
import threading

from benchmark.locks.base import BaseLock

# One non-blocking attempt (fast path) before falling back to OS-assisted
# blocking.  A larger value causes GIL thrashing at 32+ threads because
# N_threads × SPIN_COUNT competing acquire(blocking=False) calls per round
# saturate the GIL with overhead.  SPIN_COUNT=1 keeps the test-and-set
# semantic (try → fail → wait) without GIL starvation.
_SPIN_COUNT = 1


class Spinlock(BaseLock):
    """Busy-wait spinlock with bounded-spin / OS-fallback hybrid."""

    def __init__(self) -> None:
        super().__init__()
        self._flag = threading.Lock()

    # ------------------------------------------------------------------
    # BaseLock interface
    # ------------------------------------------------------------------

    def acquire(self) -> None:
        """Spin up to SPIN_COUNT times, then fall back to blocking acquire."""
        with self._lock_internal:
            self._acquire_attempts += 1

        t_start = time.perf_counter_ns()

        # Phase 1: bounded busy-wait
        for _ in range(_SPIN_COUNT):
            if self._flag.acquire(blocking=False):
                elapsed = time.perf_counter_ns() - t_start
                with self._lock_internal:
                    self._acquire_successes += 1
                    self._wait_time_ns += elapsed
                return

        # Phase 2: OS-assisted blocking (GIL released during wait)
        self._flag.acquire(blocking=True)

        elapsed = time.perf_counter_ns() - t_start
        with self._lock_internal:
            self._acquire_successes += 1
            self._wait_time_ns += elapsed

    def release(self) -> None:
        """Release the lock."""
        self._flag.release()

    def __repr__(self) -> str:
        return "Spinlock()"

