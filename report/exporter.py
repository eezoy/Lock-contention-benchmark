"""
JSON and CSV export for benchmark results.
"""

from __future__ import annotations

import csv
import json
import os
from typing import List

from benchmark.metrics import BenchmarkResult


def export_json(results: List[BenchmarkResult], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "results": [r.to_dict() for r in results],
        "summary": {
            "total_configs": len(results),
            "mechanisms": sorted({r.mechanism for r in results}),
            "thread_counts": sorted({r.thread_count for r in results}),
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def export_csv(results: List[BenchmarkResult], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "mechanism", "thread_count", "increments",
        "mean_throughput", "stddev_throughput",
        "mean_wall_time", "stddev_wall_time",
        "latency_ns", "correct",
        "contention_pct", "overhead_pct", "mean_cpu",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = r.to_dict()
            writer.writerow({k: row[k] for k in fieldnames})