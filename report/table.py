"""
ASCII table renderer with optional colorama colouring.

Columns: Mechanism | Threads | Throughput | Latency | Correct? | Contention%
"""

from __future__ import annotations

from typing import List

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    _COLOR = True
except ImportError:
    _COLOR = False

from benchmark.metrics import BenchmarkResult

# Column widths
_COL_WIDTHS = [16, 9, 18, 16, 10, 14, 12]
_HEADERS = ["Mechanism", "Threads", "Throughput(ops/s)", "Latency(ns)", "Correct?",
            "Contention%", "Overhead%"]

_MECHANISM_ORDER = [
    "threading.Lock",
    "spinlock",
    "ticket_lock",
    "mcs_lock",
    "rw_lock",
    "no_lock",
]


def _color(text: str, color_code: str) -> str:
    if not _COLOR:
        return text
    return f"{color_code}{text}{Style.RESET_ALL}"


def _fmt_row(cells: List[str]) -> str:
    return "| " + " | ".join(
        c.ljust(_COL_WIDTHS[i]) for i, c in enumerate(cells)
    ) + " |"


def _separator() -> str:
    return "+" + "+".join("-" * (_COL_WIDTHS[i] + 2) for i in range(len(_COL_WIDTHS))) + "+"


def render_table(results: List[BenchmarkResult]) -> str:
    lines: List[str] = []

    # Title
    title = "  LOCK CONTENTION BENCHMARK RESULTS"
    if _COLOR:
        title = _color(title, Fore.CYAN + Style.BRIGHT)
    lines.append(title)
    lines.append("")

    sep = _separator()
    header_row = _fmt_row(_HEADERS)
    if _COLOR:
        header_row = _color(header_row, Style.BRIGHT)

    lines.append(sep)
    lines.append(header_row)
    lines.append(sep)

    # Sort results by mechanism order, then thread count
    def _sort_key(r: BenchmarkResult):
        idx = _MECHANISM_ORDER.index(r.mechanism) if r.mechanism in _MECHANISM_ORDER else 99
        return (idx, r.thread_count)

    for r in sorted(results, key=_sort_key):
        mech = r.mechanism
        threads = str(r.thread_count)
        tput = f"{r.throughput:,.0f}"
        latency = f"{r.latency_ns:,.1f}"
        correct_str = "YES" if r.correct else "NO "
        contention = f"{r.contention_pct:.1f}"
        overhead = f"{r.overhead_pct:.1f}"

        cells = [mech, threads, tput, latency, correct_str, contention, overhead]
        row = _fmt_row(cells)

        if _COLOR:
            if not r.correct:
                row = _color(row, Fore.RED)
            elif r.contention_pct > 50:
                row = _color(row, Fore.YELLOW)
            elif r.mechanism == "no_lock":
                row = _color(row, Fore.MAGENTA)
            else:
                row = _color(row, Fore.GREEN)

        lines.append(row)

    lines.append(sep)

    # GIL warning footer
    gil_msg = (
        "\n  [!] Python GIL NOTE: CPU-bound threads cannot truly run in parallel.\n"
        "      Results show lock overhead differences, not raw parallel scaling.\n"
        "      Use --increments 1000000 to amplify lock overhead signal.\n"
        "      For true parallelism comparison, see C/pthreads implementations.\n"
    )
    if _COLOR:
        gil_msg = _color(gil_msg, Fore.YELLOW)
    lines.append(gil_msg)

    return "\n".join(lines)
