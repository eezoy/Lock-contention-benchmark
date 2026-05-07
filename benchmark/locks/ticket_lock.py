"""
Ticket Lock — FIFO-fair lock.

Algorithm
---------
Two shared counters: *ticket* (next ticket to dispense) and *serving*
(ticket currently being served).

Acquire:
  1. Atomically fetch-and-increment `ticket`  → this thread's "ticket number".
  2. Wait until `serving == my_ticket`.

Release:
  Increment `serving` by 1 → directly wake the next waiter via threading.Event.

CPython adaptation
------------------
A pure spin-wait (even with time.sleep(0)) is impractical in Python because
time.sleep(0) on Windows has a ~15 ms minimum sleep, causing O(N²) wait times
at high thread counts.  Instead each waiter registers a threading.Event; the
release path calls event.set() on exactly the next ticket's event — O(1)
direct wake-up, zero polling, algorithm semantics fully preserved.
"""

from __future__ import annotations

import time
import threading
from typing import Dict

from benchmark.locks.base import BaseLock


class TicketLock(BaseLock):
    """FIFO-fair ticket lock with Event-based waiting."""

    def __init__(self) -> None:
        super().__init__()
        self._ticket: int = 0
        self._serving: int = 0
        self._ticket_mutex = threading.Lock()
        # Map ticket_number → Event for waiting threads
        self._waiters: Dict[int, threading.Event] = {}

    # ------------------------------------------------------------------
    # BaseLock interface
    # ------------------------------------------------------------------

    def acquire(self) -> None:
        """Draw a ticket; wait on a private Event until it is called."""
        with self._lock_internal:
            self._acquire_attempts += 1

        with self._ticket_mutex:
            my_ticket = self._ticket
            self._ticket += 1
            # If it's already our turn, no need to register a waiter
            if self._serving == my_ticket:
                self._acquire_successes += 1
                return
            event = threading.Event()
            self._waiters[my_ticket] = event

        t_start = time.perf_counter_ns()
        event.wait()          # block until release() calls event.set()
        elapsed = time.perf_counter_ns() - t_start

        with self._lock_internal:
            self._acquire_successes += 1
            self._wait_time_ns += elapsed

    def release(self) -> None:
        """Advance serving and wake the next waiter directly."""
        with self._ticket_mutex:
            self._serving += 1
            next_event = self._waiters.pop(self._serving, None)
        if next_event is not None:
            next_event.set()

    def __repr__(self) -> str:
        return "TicketLock()"
