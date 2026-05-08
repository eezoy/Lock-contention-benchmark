"""
Matplotlib chart generation.

Produces 4 PNG files:
1. throughput_vs_threads.png  — line chart per mechanism
2. latency_vs_threads.png     — mean acquisition latency
3. contention_heatmap.png     — mechanism × thread_count, colour = throughput
4. overhead_breakdown.png     — stacked bar: work vs wait vs overhead
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Dict, List

from benchmark.metrics import BenchmarkResult

logger = logging.getLogger(__name__)

_MECHANISM_ORDER = [
    "threading.Lock",
    "spinlock",
    "ticket_lock",
    "mcs_lock",
    "rw_lock",
    "no_lock",
]

_COLORS = {
    "threading.Lock": "#2196F3",
    "spinlock":        "#F44336",
    "ticket_lock":     "#4CAF50",
    "mcs_lock":        "#FF9800",
    "rw_lock":         "#9C27B0",
    "no_lock":         "#607D8B",
}


def _ensure_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        logger.warning("matplotlib not installed — skipping chart generation.")
        return False


def _group_by_mechanism(
    results: List[BenchmarkResult],
) -> Dict[str, List[BenchmarkResult]]:
    grouped: Dict[str, List[BenchmarkResult]] = defaultdict(list)
    for r in results:
        grouped[r.mechanism].append(r)
    # Sort each group by thread count
    for k in grouped:
        grouped[k].sort(key=lambda r: r.thread_count)
    return grouped


def save_all_charts(results: List[BenchmarkResult], output_dir: str) -> None:
    if not _ensure_matplotlib():
        return
    os.makedirs(output_dir, exist_ok=True)
    grouped = _group_by_mechanism(results)
    _plot_throughput(grouped, output_dir)
    _plot_latency(grouped, output_dir)
    _plot_heatmap(results, output_dir)
    _plot_overhead(grouped, output_dir)
    logger.info("Charts saved to %s", output_dir)


def _plot_throughput(
    grouped: Dict[str, List[BenchmarkResult]], output_dir: str
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))

    mechanisms = [m for m in _MECHANISM_ORDER if m in grouped]
    for mech in mechanisms:
        rlist = grouped[mech]
        xs = [r.thread_count for r in rlist]
        ys = [r.throughput / 1e6 for r in rlist]          # M ops/sec
        errs = [r.stats.stddev_throughput / 1e6 for r in rlist]
        color = _COLORS.get(mech, None)
        ax.errorbar(xs, ys, yerr=errs, label=mech, marker="o",
                    linewidth=2, capsize=4, color=color)

    ax.set_title("Throughput vs Thread Count", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Throughput (M ops/sec)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([r.thread_count for rlist in grouped.values() for r in rlist])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.annotate(
        "Note: Python GIL limits true parallelism for CPU-bound threads",
        xy=(0.01, 0.01), xycoords="axes fraction", fontsize=8,
        color="grey", style="italic",
    )

    path = os.path.join(output_dir, "throughput_vs_threads.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def _plot_latency(
    grouped: Dict[str, List[BenchmarkResult]], output_dir: str
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    mechanisms = [m for m in _MECHANISM_ORDER if m in grouped]
    for mech in mechanisms:
        rlist = grouped[mech]
        xs = [r.thread_count for r in rlist]
        ys = [r.latency_ns for r in rlist]
        color = _COLORS.get(mech, None)
        ax.plot(xs, ys, label=mech, marker="s", linewidth=2, color=color)

    ax.set_title("Per-Operation Latency vs Thread Count", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Latency (ns / op)")
    ax.set_xscale("log", base=2)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "latency_vs_threads.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def _plot_heatmap(results: List[BenchmarkResult], output_dir: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    mechanisms = [m for m in _MECHANISM_ORDER
                  if any(r.mechanism == m for r in results)]
    thread_counts = sorted({r.thread_count for r in results})

    data = np.zeros((len(mechanisms), len(thread_counts)))
    for r in results:
        if r.mechanism in mechanisms:
            mi = mechanisms.index(r.mechanism)
            ti = thread_counts.index(r.thread_count)
            data[mi, ti] = r.throughput / 1e6  # M ops/sec

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Throughput (M ops/sec)")

    ax.set_xticks(range(len(thread_counts)))
    ax.set_xticklabels([str(t) for t in thread_counts])
    ax.set_yticks(range(len(mechanisms)))
    ax.set_yticklabels(mechanisms)
    ax.set_xlabel("Thread Count")
    ax.set_title("Throughput Heatmap (Mechanism × Thread Count)", fontsize=13, fontweight="bold")

    # Annotate cells
    for mi in range(len(mechanisms)):
        for ti in range(len(thread_counts)):
            val = data[mi, ti]
            ax.text(ti, mi, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="black")

    path = os.path.join(output_dir, "contention_heatmap.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def _plot_overhead(
    grouped: Dict[str, List[BenchmarkResult]], output_dir: str
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    mechanisms = [m for m in _MECHANISM_ORDER if m in grouped]
    # Use the highest thread count for comparison
    reps: List[BenchmarkResult] = []
    for mech in mechanisms:
        rlist = grouped[mech]
        reps.append(max(rlist, key=lambda r: r.thread_count))

    labels = [r.mechanism for r in reps]
    work_pct = [max(0.0, 100.0 - r.overhead_pct) for r in reps]
    wait_pct = [r.overhead_pct for r in reps]

    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_work = ax.bar(x, work_pct, width, label="Work time", color="#4CAF50", alpha=0.85)
    bars_wait = ax.bar(x, wait_pct, width, bottom=work_pct, label="Lock wait time",
                       color="#F44336", alpha=0.85)

    ax.set_title("Lock Overhead Breakdown (highest thread count)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Time percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    path = os.path.join(output_dir, "overhead_breakdown.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)