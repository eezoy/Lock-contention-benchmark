# Lock Contention Benchmark

A **real benchmark suite** that measures the scalability and overhead of six different locking mechanisms under contention in Python. The project studies how throughput, latency, and lock overhead change as the number of concurrent threads grows from 1 to 64+.

> Built for the *Operating Systems* / *Computer Architecture* course — demonstrates synchronisation primitives, cache effects, and the impact of Python's GIL on CPU-bound parallel workloads.

---

## What it does and why it matters

Modern CPUs are multi-core, and correct concurrent programming requires synchronisation. Different locking algorithms make very different trade-offs:

| Goal | Relevant lock |
|---|---|
| Lowest single-thread overhead | `threading.Lock` (native mutex) |
| Visible busy-wait cost | Spinlock |
| FIFO fairness guarantees | Ticket Lock |
| Cache-friendly NUMA scalability | MCS Lock |
| Read-heavy workloads | RW-Lock |
| Theoretical ceiling (unsynchronised) | No lock (baseline) |

This suite makes those trade-offs **measurable and visible** through terminal tables and four matplotlib charts.

---

## Install

```bash
pip install -r requirements.txt
```

**Dependencies:** `matplotlib`, `numpy`, `colorama`, `psutil`  
**Python:** 3.10+

---

## Run

```bash
# Full benchmark — all mechanisms, 7 thread counts, saves plots
python main.py

# Custom thread counts
python main.py --threads 1,2,4,8,16,32

# Increase operations to make lock overhead more visible
python main.py --increments 500000

# More repetitions for tighter statistics
python main.py --repeat 10

# Benchmark a single mechanism
python main.py --mechanism spinlock

# Custom output directory
python main.py --output my_results/

# Suppress chart generation (CI / headless)
python main.py --no-plots

# Verbose logging
python main.py --verbose
```

---

## Sample terminal output

```
======================================================================
  LOCK CONTENTION BENCHMARK SUITE
  Python 3.14.0  |  Windows AMD64
  Threads: [1, 2, 4, 8, 16, 32, 64]
  Increments per thread: 20,000
  Repetitions: 5
======================================================================

Running correctness validation …
  threading.Lock       PASS
  spinlock             PASS
  ticket_lock          PASS
  mcs_lock             PASS
  rw_lock              PASS

  LOCK CONTENTION BENCHMARK RESULTS

+------------------+-----------+--------------------+------------------+------------+----------------+--------------+
| Mechanism        | Threads   | Throughput(ops/s)  | Latency(ns)      | Correct?   | Contention%    | Overhead%    |
+------------------+-----------+--------------------+------------------+------------+----------------+--------------+
| threading.Lock   | 1         | 1,444,606          | 692.2            | YES        | 0.0            | 71.2         |
| threading.Lock   | 4         | 1,394,732          | 717.0            | YES        | 300.0          | 28.5         |
| spinlock         | 1         | 1,212,139          | 825.0            | YES        | 0.0            | 75.0         |
| ...              | ...       | ...                | ...              | ...        | ...            | ...          |
| no_lock          | 64        | 2,400,000          | 416.7            | YES        | 0.0            | 5.0          |
+------------------+-----------+--------------------+------------------+------------+----------------+--------------+

  [!] Python GIL NOTE: CPU-bound threads cannot truly run in parallel.
      Use --increments 1000000 to amplify lock overhead signal.
```

---

## Project structure

```
lock-contention-benchmark/
├── main.py                      # CLI entry point — python main.py
├── requirements.txt
├── benchmark/
│   ├── runner.py                # BenchmarkRunner — harness + timing
│   ├── metrics.py               # BenchmarkConfig / BenchmarkResult / RunStats
│   └── locks/
│       ├── base.py              # Abstract BaseLock (acquire/release/context-manager)
│       ├── spinlock.py          # Busy-wait spinlock
│       ├── ticket_lock.py       # FIFO ticket lock
│       ├── mcs_lock.py          # Queue-based MCS lock
│       └── rw_lock.py           # Readers-writer lock
├── report/
│   ├── table.py                 # Coloured ASCII table (colorama)
│   ├── charts.py                # 4 × matplotlib PNG charts
│   └── exporter.py              # JSON + CSV export
└── results/                     # Auto-created, gitignored
    ├── benchmark_results.json
    ├── benchmark_results.csv
    └── plots/
        ├── throughput_vs_threads.png
        ├── latency_vs_threads.png
        ├── contention_heatmap.png
        └── overhead_breakdown.png
```

---

## Lock mechanisms — theory

### 1. `threading.Lock` — native mutex
CPython's built-in lock wraps the OS-level pthread mutex (on Linux) or
`CRITICAL_SECTION` / `SRWLOCK` (on Windows). Kernel involvement means the
OS can put a blocked thread to sleep, freeing the CPU — but that involves
a syscall.  **Best general-purpose choice.**

### 2. Spinlock — busy-wait
Instead of sleeping, a waiting thread loops, repeatedly calling
`acquire(blocking=False)` until it succeeds.  Low latency when the lock is
held briefly; **wastes CPU cycles** under high contention.  No fairness —
threads acquire in random order.  In Python `time.sleep(0)` is added to
each spin to yield the GIL and avoid starving the lock holder.

### 3. Ticket Lock — FIFO spinlock
Two counters: `ticket` (next ticket issued) and `serving` (current holder).
Acquire atomically grabs a ticket; threads spin until `serving == my_ticket`.
**Strictly FIFO** — the first thread to call acquire is the first to enter.
Eliminates the "thundering herd" of unfair spinlocks.  All threads still
watch the same `serving` cache line (unlike MCS).

### 4. MCS Lock — queue-based, cache-friendly
Each thread owns a private node with a `locked` flag and a `next` pointer.
On acquire, the thread appends its node to a global queue and **spins on its
own node** — not a shared variable.  The releasing thread sets the
successor's `locked = False`, waking exactly one waiter.  On NUMA hardware,
this eliminates cache-line bouncing: no false sharing, no broadcast
invalidation.  **Best choice for large multi-socket systems** (in C/pthreads;
GIL limits the advantage in Python).

### 5. RW-Lock — shared readers, exclusive writer
Multiple threads may hold the *read* lock concurrently; a writer requires
exclusive access.  Writers are given priority over new readers to prevent
writer starvation.  In this benchmark every thread does a read-modify-write
(increment), so it always acquires the *write* lock — the RWLock degenerates
to a mutex for this workload, showing its state-machine overhead vs a plain
`threading.Lock`.

### 6. No lock — unsynchronised baseline
Performs increments with **no synchronisation**.  Demonstrates race
conditions (final counter may not equal expected value) and shows the
**theoretical throughput ceiling** without any locking overhead.

---

## How to interpret the plots

### `throughput_vs_threads.png`
Each line is one mechanism.  In a perfect parallel system, throughput would
scale linearly with thread count.  Python's GIL prevents this for CPU-bound
work — you will see throughput **plateau or even fall** at 2+ threads.
`no_lock` shows the GIL ceiling.  The gap between `no_lock` and any locked
mechanism is the **cost of synchronisation**.

### `latency_vs_threads.png`
Per-operation latency rises with thread count for contended locks.  Locks
with more complex acquire paths (MCS, RWLock) show higher base latency
at 1 thread; simpler locks (threading.Lock, spinlock) have lower base
latency but may grow faster under contention.

### `contention_heatmap.png`
Colour encodes throughput across the full (mechanism × thread-count) matrix.
Hot cells (bright) = high throughput.  On real C benchmarks, MCS would stay
hot at high thread counts; in Python the GIL compresses all values.

### `overhead_breakdown.png`
Stacked bar at the highest thread count.  Green = time threads spent doing
useful work; red = time spent waiting for the lock.  Mechanisms with large
red bars have high lock overhead under contention.

---

## Python GIL — important note

CPython's **Global Interpreter Lock (GIL)** ensures that only one thread
executes Python bytecode at a time.  This has two major consequences for
this benchmark:

1. **True parallel scaling is not visible.**  A 16-thread run will not be
   16× faster than a 1-thread run.  The GIL serialises all threads anyway.

2. **Busy-wait locks behave differently.**  A spinning thread holds the GIL,
   preventing the lock *holder* from running.  The `time.sleep(0)` call in
   each spin loop explicitly releases the GIL so other threads can progress.

Despite these constraints, the benchmark **does** reveal meaningful data:

- Relative overhead of each lock's acquire/release path.
- How contention events scale with thread count.
- Correctness of the lock implementations.
- The overhead cost of condition variables (RWLock) vs simple atomics.

For raw parallel speedup numbers, reimplement these algorithms in C with
pthreads and compare with `perf stat`.

**Recommended settings to amplify lock-overhead signal:**
```bash
python main.py --increments 1000000 --threads 1,2,4,8
```

---

## Known limitations

- **Python GIL** prevents true CPU parallelism (see above).
- **Spinlock / TicketLock**: `time.sleep(0)` is used to yield the GIL; this
  changes the spin-wait character compared to a hardware CAS loop in C.
- **MCS node reuse**: thread-local nodes are reused across runs; this is
  correct but means the benchmark does not measure allocation cost.
- **`threading.Lock` CAS emulation**: Python has no native compare-and-swap;
  all "atomic" operations are serialised by a stdlib lock.
- `no_lock` counter races may result in the correct final count on CPython
  (due to the GIL making `+=` quasi-atomic), so `Correct? = YES` for
  `no_lock` is expected but not guaranteed on non-CPython runtimes.
- CPU usage via `psutil` measures the whole process, not per-thread CPU time.
