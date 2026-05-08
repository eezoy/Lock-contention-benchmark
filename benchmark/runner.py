"""
BenchmarkRunner — core harness that times each locking mechanism.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Type

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

from benchmark.locks.base import BaseLock
from benchmark.locks.spinlock import Spinlock
from benchmark.locks.ticket_lock import TicketLock
from benchmark.locks.mcs_lock import MCSLock
from benchmark.locks.rw_lock import RWLock
from benchmark.metrics import BenchmarkConfig, BenchmarkResult, RunStats

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Registry of all lock types
# ------------------------------------------------------------------

LOCK_REGISTRY: Dict[str, Callable[[], BaseLock]] = {
    "threading.Lock": threading.Lock,   # stdlib — used via wrapper below
    "spinlock": Spinlock,
    "ticket_lock": TicketLock,
    "mcs_lock": MCSLock,
    "rw_lock": RWLock,
    "no_lock": lambda: _NoLock(),       # baseline
}


class _StdlibLockWrapper(BaseLock):
    """Thin wrapper so threading.Lock is compatible with BaseLock interface."""

    def __init__(self) -> None:
        super().__init__()
        self._inner = threading.Lock()

    def acquire(self) -> None:
        with self._lock_internal:
            self._acquire_attempts += 1
        t = time.perf_counter_ns()
        self._inner.acquire()
        elapsed = time.perf_counter_ns() - t
        with self._lock_internal:
            self._acquire_successes += 1
            self._wait_time_ns += elapsed

    def release(self) -> None:
        self._inner.release()


class _NoLock(BaseLock):
    """No-op lock — shows race conditions and ceiling throughput."""

    def acquire(self) -> None:
        with self._lock_internal:
            self._acquire_attempts += 1
            self._acquire_successes += 1

    def release(self) -> None:
        pass


# Replace the factory so we use the wrapper for threading.Lock
LOCK_REGISTRY["threading.Lock"] = _StdlibLockWrapper


# ------------------------------------------------------------------
# Worker function
# ------------------------------------------------------------------

def _worker(
    lock: BaseLock,
    counter: List[int],
    increments: int,
    barrier: threading.Barrier,
    thread_wait_ns: List[int],
    thread_idx: int,
) -> None:
    """Increment shared counter `increments` times under `lock`."""
    barrier.wait()  # synchronise start
    local_wait = 0
    for _ in range(increments):
        t0 = time.perf_counter_ns()
        lock.acquire()
        t1 = time.perf_counter_ns()
        counter[0] += 1
        lock.release()
        local_wait += t1 - t0
    thread_wait_ns[thread_idx] = local_wait


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

class BenchmarkRunner:
    """Runs benchmarks for all (or selected) lock mechanisms."""

    def __init__(
        self,
        thread_counts: List[int],
        increments: int,
        repeat: int,
        mechanisms: Optional[List[str]] = None,
    ) -> None:
        self.thread_counts = thread_counts
        self.increments = increments
        self.repeat = repeat
        self.mechanisms = mechanisms or list(LOCK_REGISTRY.keys())

        # Validate mechanism names
        unknown = [m for m in self.mechanisms if m not in LOCK_REGISTRY]
        if unknown:
            raise ValueError(f"Unknown mechanism(s): {unknown}. "
                             f"Available: {list(LOCK_REGISTRY.keys())}")

    def run_all(self) -> List[BenchmarkResult]:
        results: List[BenchmarkResult] = []
        total = len(self.mechanisms) * len(self.thread_counts)
        done = 0
        for mechanism in self.mechanisms:
            for thread_count in self.thread_counts:
                done += 1
                logger.info(
                    "[%d/%d] %s × %d threads × %d ops × %d reps",
                    done, total, mechanism, thread_count,
                    self.increments, self.repeat,
                )
                result = self._run_one(mechanism, thread_count)
                results.append(result)
        return results

    def _run_one(self, mechanism: str, thread_count: int) -> BenchmarkResult:
        config = BenchmarkConfig(
            mechanism=mechanism,
            thread_count=thread_count,
            increments=self.increments,
            repeat=self.repeat,
        )
        stats = RunStats()

        for rep in range(self.repeat):
            lock = LOCK_REGISTRY[mechanism]()
            lock.reset_counters()
            counter = [0]
            barrier = threading.Barrier(thread_count)
            thread_wait_ns: List[int] = [0] * thread_count

            # CPU usage baseline
            cpu_before: float = 0.0
            proc = None
            if _PSUTIL_AVAILABLE:
                try:
                    proc = psutil.Process()
                    proc.cpu_percent(interval=None)  # prime
                except Exception:
                    proc = None

            threads = [
                threading.Thread(
                    target=_worker,
                    args=(lock, counter, self.increments, barrier,
                          thread_wait_ns, i),
                    daemon=True,
                )
                for i in range(thread_count)
            ]

            t_start = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall_time = time.perf_counter() - t_start

            # CPU measurement
            if proc is not None:
                try:
                    cpu_usage = proc.cpu_percent(interval=None)
                except Exception:
                    cpu_usage = 0.0
            else:
                cpu_usage = 0.0

            total_ops = config.total_ops
            throughput = total_ops / wall_time if wall_time > 0 else 0.0
            correct = (counter[0] == total_ops) if mechanism != "no_lock" else True

            # Contention ratio = measured / ideal (single-thread time)
            ideal_time = self.increments / throughput if throughput > 0 else 0.0
            contention_ratio = (
                (wall_time / ideal_time - 1.0)
                if ideal_time > 0 and thread_count > 1
                else 0.0
            )
            contention_ratio = max(0.0, contention_ratio)

            stats.wall_times.append(wall_time)
            stats.throughputs.append(throughput)
            stats.correct.append(correct)
            stats.contention_ratios.append(contention_ratio)
            stats.wait_times_ns.append(sum(thread_wait_ns))
            stats.cpu_usages.append(cpu_usage)

            logger.debug(
                "  rep %d: wall=%.3fs tput=%.0f ops/s correct=%s",
                rep + 1, wall_time, throughput, correct,
            )

        return BenchmarkResult(config=config, stats=stats)