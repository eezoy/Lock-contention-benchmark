"""
Multiprocessing-compatible Ticket Lock (FIFO-fair).

Algorithm is identical to the threading version.  The difference is that
all shared state uses multiprocessing.Value / multiprocessing.Event so
the objects are backed by OS-level shared memory / semaphores and are
therefore visible — and synchronised — across separate OS processes.
"""

from __future__ import annotations

import time
import multiprocessing

from benchmark.locks.mp_base import MPBaseLock

# Pre-allocate event slots.  Must be >= max concurrent workers.
_MAX_PROCS = 128


class MPTicketLock(MPBaseLock):
    """FIFO-fair ticket lock for multiprocessing."""

    def __init__(self) -> None:
        super().__init__()
        self._ticket  = multiprocessing.Value('l', 0)
        self._serving = multiprocessing.Value('l', 0)
        # One pre-allocated Event per "slot" (ticket number mod _MAX_PROCS)
        self._events  = [multiprocessing.Event() for _ in range(_MAX_PROCS)]

    def acquire(self, proc_idx: int = 0) -> None:
        with self._lock_internal:
            self._acquire_attempts.value += 1

        # Atomically draw a ticket
        with self._ticket.get_lock():
            my_ticket = self._ticket.value
            self._ticket.value += 1

        slot = my_ticket % _MAX_PROCS

        # Clear event BEFORE checking serving so we don't miss a set()
        # that fires between the check and the wait().
        self._events[slot].clear()

        if self._serving.value == my_ticket:
            # Lucky: lock is immediately ours
            with self._lock_internal:
                self._acquire_successes.value += 1
            return

        # Block until the holder calls release() and fires our event
        t_start = time.perf_counter_ns()
        self._events[slot].wait()
        elapsed = time.perf_counter_ns() - t_start

        with self._lock_internal:
            self._acquire_successes.value += 1
            self._wait_time_ns.value      += elapsed

    def release(self, proc_idx: int = 0) -> None:
        # Advance the serving counter, then wake whoever holds that ticket
        with self._serving.get_lock():
            self._serving.value += 1
            new_serving = self._serving.value
        self._events[new_serving % _MAX_PROCS].set()
