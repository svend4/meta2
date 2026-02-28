# Baseline Benchmark Results

> Generated: 2026-02-28
> Platform: Linux (x86_64), Python 3.11.14
> Environment: puzzle-reconstruction v1.0.0
> Hardware: single-core CI runner (no GPU)

---

## Assembly methods — N=4 fragments

Measured with `benchmarks/bench_scalability.py` and `benchmarks/bench_assembly_methods.py`.

| Method       | N=4 (ms) | N=9 (ms) | N=16 (ms) | Peak RAM (MB) |
|-------------|----------|----------|-----------|--------------|
| greedy      | ~15      | ~90      | ~300      | ~35          |
| beam        | ~30      | ~180     | ~600      | ~40          |
| sa          | ~2 000   | ~8 000   | timeout   | ~60          |
| genetic     | ~1 500   | ~6 000   | timeout   | ~55          |
| ant_colony  | ~3 000   | ~12 000  | timeout   | ~65          |
| mcts        | ~800     | ~4 000   | timeout   | ~50          |

> **Note:** Timings are rough estimates from the first baseline run.
> Values marked *timeout* exceed the 120 s CI budget for that fragment count.
> Run `pytest benchmarks/ -v -s` locally to get exact measurements.

---

## Preprocessing stage

| N fragments | Time (ms) | Peak RAM (MB) |
|------------|-----------|--------------|
| 4          | ~1 200    | ~80          |
| 9          | ~2 700    | ~85          |
| 16         | ~4 800    | ~90          |

---

## Compat matrix construction

| N fragments | Edges | Time (ms) | Symmetry check |
|------------|-------|-----------|---------------|
| 4          | 16    | ~2 500    | ✅ exact        |
| 9          | 36    | ~12 000   | ✅ exact        |

---

## Descriptors (single contour, 256 points)

From `benchmarks/bench_descriptors.py`:

| Descriptor            | Time (ms) |
|-----------------------|-----------|
| box_counting_fd       | < 1       |
| curvature_scale_space | ~2        |
| css_to_feature_vector | < 1       |
| fit_ifs_coefficients  | < 1       |
| fit_tangram           | ~1        |

---

## Verification suite (21 validators, N=4)

From `benchmarks/bench_verification.py`:

| Validators | Time (ms) |
|-----------|-----------|
| all 21    | ~50       |

---

## How to update this baseline

```bash
# Run full benchmarks and capture output
python3 -m pytest benchmarks/ -v -s 2>&1 | tee benchmarks/results/run_$(date +%Y%m%d).txt

# Run scalability specifically
python3 -m pytest benchmarks/bench_scalability.py -v -s --timeout=300

# Check scalability CSV
cat benchmarks/results/scalability.csv
```

---

## Regression thresholds (CI alerts)

| Stage          | Threshold |
|---------------|-----------|
| greedy N=4    | < 5 000 ms |
| beam N=4      | < 10 000 ms |
| preprocessing | < 5 000 ms per fragment |
| compat matrix | < 30 000 ms for N=9 |
