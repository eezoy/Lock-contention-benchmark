#!/usr/bin/env python3
"""
lock-contention-benchmark — main entry point.

Usage:
    python main.py
    python main.py --threads 1,2,4,8,16,32
    python main.py --increments 500000
    python main.py --repeat 5
    python main.py --mechanism spinlock
    python main.py --output results/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import platform

from benchmark.runner import (
    BenchmarkRunner, LOCK_REGISTRY,
    MultiprocessingBenchmarkRunner, MP_LOCK_REGISTRY,
)
from report.table import render_table
from report.charts import save_all_charts
from report.exporter import export_json, export_csv

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("main")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lock-contention benchmark suite — measure locking scalability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--threads",
        default="1,2,4,8,16,32,64",
        help="Comma-separated thread counts to test (default: 1,2,4,8,16,32,64)",
    )
    parser.add_argument(
        "--increments",
        type=int,
        default=2_000,
        help="Shared-counter increments per thread per run (default: 2000). "
             "Use --increments 100000 for publication-quality results (takes longer).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Repetitions per configuration for statistical averaging (default: 3)",
    )
    parser.add_argument(
        "--mechanism",
        default=None,
        help=(
            "Run only one mechanism. Choices: "
            + ", ".join(LOCK_REGISTRY.keys())
        ),
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Directory for JSON, CSV, and PNG outputs (default: results/)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip chart generation (useful in headless CI environments)",
    )
    parser.add_argument(
        "--backend",
        default="threading",
        choices=["threading", "multiprocessing"],
        help=(
            "Concurrency backend. \'threading\' (default) uses threads and is "
            "limited by the GIL. \'multiprocessing\' spawns real OS processes "
            "for true parallelism and realistic lock-contention measurements."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Validation tests (quick sanity checks run before benchmarks)
# ------------------------------------------------------------------

def _run_validation() -> bool:
    """
    Verify correctness of every lock by running a small concurrent increment.
    Returns True if all pass.
    """
    from benchmark.runner import LOCK_REGISTRY
    import threading

    THREADS = 8
    INCREMENTS = 1_000
    passed = True

    print("Running correctness validation …", flush=True)
    for name, factory in LOCK_REGISTRY.items():
        if name == "no_lock":
            continue  # no_lock intentionally races
        lock = factory()
        counter = [0]
        barrier = threading.Barrier(THREADS)

        def _w(lk=lock, ctr=counter, b=barrier):
            b.wait()
            for _ in range(INCREMENTS):
                with lk:
                    ctr[0] += 1

        ts = [threading.Thread(target=_w, daemon=True) for _ in range(THREADS)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

        expected = THREADS * INCREMENTS
        ok = counter[0] == expected
        status = "PASS" if ok else f"FAIL (got {counter[0]}, expected {expected})"
        print(f"  {name:<20} {status}")
        if not ok:
            passed = False

    print()
    return passed


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse thread counts
    try:
        thread_counts = [int(x.strip()) for x in args.threads.split(",")]
    except ValueError:
        logger.error("--threads must be comma-separated integers, e.g. 1,2,4,8")
        return 1

    mechanisms = [args.mechanism] if args.mechanism else None

    # ------------------------------------------------------------------
    # Print banner
    # ------------------------------------------------------------------
    is_mp = args.backend == "multiprocessing"
    active_registry = MP_LOCK_REGISTRY if is_mp else LOCK_REGISTRY

    print("=" * 70)
    print("  LOCK CONTENTION BENCHMARK SUITE")
    print(f"  Python {sys.version.split()[0]}  |  {platform.system()} {platform.machine()}")
    print(f"  Backend:  {args.backend.upper()}")
    print(f"  {'Processes' if is_mp else 'Threads'}: {thread_counts}")
    print(f"  Increments per {'process' if is_mp else 'thread'}: {args.increments:,}")
    print(f"  Repetitions: {args.repeat}")
    print(f"  Mechanisms: {mechanisms or list(active_registry.keys())}")
    print("=" * 70)
    print()

    if is_mp:
        print(
            "  MULTIPROCESSING NOTE:\n"
            "  Using real OS processes — GIL is bypassed. Each process runs on\n"
            "  its own CPU core, producing realistic lock-contention measurements.\n"
            "  Spinlock now burns real CPU cycles; MCS/Ticket show FIFO fairness.\n"
        )
    else:
        print(
            "  Python GIL NOTE:\n"
            "  CPython's Global Interpreter Lock (GIL) prevents true parallel\n"
            "  execution of CPU-bound threads. This benchmark measures lock\n"
            "  *overhead* and contention *patterns* — not raw parallel speedup.\n"
            "  Use --backend multiprocessing for realistic results.\n"
        )

    # ------------------------------------------------------------------
    # Correctness validation
    # ------------------------------------------------------------------
    if not _run_validation():
        logger.error("Validation failed — aborting benchmark.")
        return 2

    # ------------------------------------------------------------------
    # Run benchmarks
    # ------------------------------------------------------------------
    try:
        if is_mp:
            runner = MultiprocessingBenchmarkRunner(
                thread_counts=thread_counts,
                increments=args.increments,
                repeat=args.repeat,
                mechanisms=mechanisms,
            )
        else:
            runner = BenchmarkRunner(
                thread_counts=thread_counts,
                increments=args.increments,
                repeat=args.repeat,
                mechanisms=mechanisms,
            )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Starting benchmark runs …")
    results = runner.run_all()
    logger.info("Benchmark runs complete.")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    output_dir = args.output
    plots_dir = os.path.join(output_dir, "plots")
    json_path = os.path.join(output_dir, "benchmark_results.json")
    csv_path = os.path.join(output_dir, "benchmark_results.csv")

    os.makedirs(output_dir, exist_ok=True)

    # Terminal table
    print(render_table(results))

    # JSON + CSV
    export_json(results, json_path)
    export_csv(results, csv_path)
    logger.info("Results saved → %s, %s", json_path, csv_path)

    # Plots
    if not args.no_plots:
        save_all_charts(results, plots_dir)
    else:
        logger.info("Plot generation skipped (--no-plots).")

    print(f"\nAll output saved to: {os.path.abspath(output_dir)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
