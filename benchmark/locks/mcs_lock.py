"""
MCS Lock — queue-based, cache-friendly lock.

Algorithm (Mellor-Crummey & Scott, 1991)
-----------------------------------------
Each thread owns a *node* with a `locked` field and a `next` pointer.
The lock itself is just a tail pointer.

Acquire:
  1. Set node.locked = True, node.next = None.
  2. Atomically swap the tail with our node → get predecessor.
  3. If no predecessor: lock was free, done.
  4. Otherwise: link predecessor.next = our_node and wait on node.event.

Release:
  1. If no successor: atomically set tail = None, done.
  2. Otherwise: wait for node.next to be set, then wake successor via event.

CPython adaptation
------------------
Spinning on node.locked with time.sleep(0) is impractical on Windows
(~15 ms minimum sleep).  Each node carries a threading.Event; the release
path calls successor.event.set() for a direct O(1) wake-up.  Algorithm
structure and FIFO semantics are fully preserved.
"""

from __future__ import annotations

import time
import threading
from typing import Optional

from benchmark.locks.base import BaseLock


class _MCSNode:
    """Per-thread queue node."""

    __slots__ = ("event", "next")

    def __init__(self) -> None:
        self.event: threading.Event = threading.Event()
        self.next: Optional["_MCSNode"] = None


_thread_local = threading.local()


class MCSLock(BaseLock):
    """MCS queue-based cache-friendly lock."""

    def __init__(self) -> None:
        super().__init__()
        self._tail: Optional[_MCSNode] = None
        self._tail_mutex = threading.Lock()

    # ------------------------------------------------------------------
    # BaseLock interface
    # ------------------------------------------------------------------

    def acquire(self) -> None:
        """Enqueue this thread's node and wait on its private Event."""
        with self._lock_internal:
            self._acquire_attempts += 1

        if not hasattr(_thread_local, "node"):
            _thread_local.node = _MCSNode()
        node: _MCSNode = _thread_local.node
        node.event.clear()
        node.next = None

        t_start = time.perf_counter_ns()

        with self._tail_mutex:
            predecessor = self._tail
            self._tail = node

        if predecessor is not None:
            predecessor.next = node
            node.event.wait()   # block until predecessor calls event.set()

        elapsed = time.perf_counter_ns() - t_start
        with self._lock_internal:
            self._acquire_successes += 1
            self._wait_time_ns += elapsed

    def release(self) -> None:
        """Dequeue and wake the successor if any."""
        node: _MCSNode = _thread_local.node

        if node.next is None:
            with self._tail_mutex:
                if self._tail is node:
                    self._tail = None
                    return
                # A new waiter is linking — wait for node.next to be set
            # Busy-wait here is bounded: only one iteration of linking
            while node.next is None:
                pass  # very brief — successor sets next then blocks on event

        node.next.event.set()

    def __repr__(self) -> str:
        return "MCSLock()"
