"""
Dataclasses for benchmark configuration and results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkConfig:
    """Parameters for a single benchmark run configuration."""

    mechanism: str          # human-readable lock name
    thread_count: int       # number of concurrent threads
    increments: int         # operations per thread
    repeat: int = 5         # repetitions for statistical stability

    @property
    def total_ops(self) -> int:
        return self.thread_count * self.increments


@dataclass
class RunStats:
    """Statistics computed over multiple repetitions of one configuration."""

    wall_times: List[float] = field(default_factory=list)       # seconds
    throughputs: List[float] = field(default_factory=list)      # ops/sec
    correct: List[bool] = field(default_factory=list)
    contention_ratios: List[float] = field(default_factory=list)
    wait_times_ns: List[int] = field(default_factory=list)      # total wait ns
    cpu_usages: List[float] = field(default_factory=list)       # percent

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------

    def mean(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def stddev(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        m = self.mean(values)
        variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    @property
    def mean_wall_time(self) -> float:
        return self.mean(self.wall_times)

    @property
    def stddev_wall_time(self) -> float:
        return self.stddev(self.wall_times)

    @property
    def mean_throughput(self) -> float:
        return self.mean(self.throughputs)

    @property
    def stddev_throughput(self) -> float:
        return self.stddev(self.throughputs)

    @property
    def all_correct(self) -> bool:
        return all(self.correct)

    @property
    def mean_contention_ratio(self) -> float:
        return self.mean(self.contention_ratios)

    @property
    def mean_wait_time_ns(self) -> float:
        return self.mean([float(w) for w in self.wait_times_ns])

    @property
    def mean_cpu(self) -> float:
        return self.mean(self.cpu_usages)


@dataclass
class BenchmarkResult:
    """Aggregated result for one (mechanism, thread_count) configuration."""

    config: BenchmarkConfig
    stats: RunStats

    @property
    def mechanism(self) -> str:
        return self.config.mechanism

    @property
    def thread_count(self) -> int:
        return self.config.thread_count

    @property
    def throughput(self) -> float:
        return self.stats.mean_throughput

    @property
    def latency_ns(self) -> float:
        """Mean per-operation latency in nanoseconds."""
        if self.throughput == 0:
            return 0.0
        return 1e9 / self.throughput

    @property
    def correct(self) -> bool:
        return self.stats.all_correct

    @property
    def contention_pct(self) -> float:
        return self.stats.mean_contention_ratio * 100

    @property
    def overhead_pct(self) -> float:
        """Fraction of total time threads spent waiting for the lock."""
        total_work_ns = self.stats.mean_wall_time * 1e9 * self.config.thread_count
        if total_work_ns == 0:
            return 0.0
        return min(100.0, self.stats.mean_wait_time_ns / total_work_ns * 100)
    def to_dict(self) -> dict:
        return {
            "mechanism": self.mechanism,
            "thread_count": self.thread_count,
            "increments": self.config.increments,
            "mean_throughput": self.stats.mean_throughput,
            "stddev_throughput": self.stats.stddev_throughput,
            "mean_wall_time": self.stats.mean_wall_time,
            "stddev_wall_time": self.stats.stddev_wall_time,
            "latency_ns": self.latency_ns,
            "correct": self.correct,
            "contention_pct": self.contention_pct,
            "overhead_pct": self.overhead_pct,
            "mean_cpu": self.stats.mean_cpu,
            "wall_times": self.stats.wall_times,
            "throughputs": self.stats.throughputs,
        }