"""
Multiprocessing-compatible MCS Lock (Mellor-Crummey & Scott, 1991).

Each process uses its proc_idx as a pre-allocated queue node slot.
All inter-process state (tail pointer, next[] array, events[]) lives in
shared memory / OS semaphores so it is visible across processes.

Slot-based adaptation
---------------------
Classic MCS uses pointers to thread-local heap nodes.  Here we replace
pointers with integer slot indices (0 … _MAX_SLOTS-1).  Each worker
process maps to a fixed slot via  proc_idx % _MAX_SLOTS.

acquire(proc_idx):
  1. Clear events[slot],  set next[slot] = -1.
  2. Atomically swap tail → slot,  get predecessor index.
  3. If no predecessor  → lock is free, done.
  4. Link  next[pred] = slot,  then wait on events[slot].

release(proc_idx):
  1. If next[slot] == -1: try to set tail = -1 (CAS-like under tail_lock).
     If another process is mid-link, busy-wait (extremely brief).
  2. Wake successor via events[next[slot]].set().
"""

from __future__ import annotations

import time
import multiprocessing

from benchmark.locks.mp_base import MPBaseLock

_MAX_SLOTS = 128


class MPMCSLock(MPBaseLock):
    """MCS queue-based cache-friendly lock for multiprocessing."""

    def __init__(self) -> None:
        super().__init__()
        # Tail index (-1 = queue empty)
        self._tail_idx = multiprocessing.Value('i', -1)
        # next[i] = successor slot index for node i  (-1 = none)
        self._next     = multiprocessing.Array('i', [-1] * _MAX_SLOTS)
        # One event per slot — used to wake a waiting process
        self._events   = [multiprocessing.Event() for _ in range(_MAX_SLOTS)]
        # Protects the tail swap (replaces the CAS instruction)
        self._tail_lock = multiprocessing.Lock()

    def acquire(self, proc_idx: int = 0) -> None:
        with self._lock_internal:
            self._acquire_attempts.value += 1

        slot = proc_idx % _MAX_SLOTS
        self._events[slot].clear()
        self._next[slot] = -1

        t_start = time.perf_counter_ns()

        # Atomic swap: set tail = slot, read previous tail
        with self._tail_lock:
            pred_idx = self._tail_idx.value
            self._tail_idx.value = slot

        if pred_idx == -1:
            # Queue was empty — lock is ours immediately
            with self._lock_internal:
                self._acquire_successes.value += 1
            return

        # Enqueue behind predecessor and wait
        self._next[pred_idx] = slot
        self._events[slot].wait()

        elapsed = time.perf_counter_ns() - t_start
        with self._lock_internal:
            self._acquire_successes.value += 1
            self._wait_time_ns.value      += elapsed

    def release(self, proc_idx: int = 0) -> None:
        slot = proc_idx % _MAX_SLOTS

        if self._next[slot] == -1:
            # No known successor — try to mark queue as empty
            with self._tail_lock:
                if self._tail_idx.value == slot:
                    self._tail_idx.value = -1
                    return
            # A successor is mid-link (set next[slot] but not yet waiting).
            # Spin — this window is a handful of nanoseconds at most.
            while self._next[slot] == -1:
                pass

        # Wake the successor
        self._events[self._next[slot]].set()
