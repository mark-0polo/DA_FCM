# DA-FCM UCI Benchmark

This folder is a clean benchmark package for the DA-FCM UCI experiments.

It runs six clustering methods on eight UCI datasets:

- FCM
- PCM
- PFCM
- DPC-FCM
- eFCM
- DA-FCM

Datasets included:

- Iris
- Wine
- Breast Cancer Wisconsin Diagnostic
- Seeds
- Glass
- Ecoli
- Yeast
- Statlog/Image Segmentation

## Quick Start

Use Python 3.13 if possible. The package was prepared with Python 3.13.5.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_benchmark.py
```

On macOS/Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Outputs

Running `python run_benchmark.py` writes:

- `results/all_algorithms_uci_benchmark_summary.csv`
- `results/all_algorithms_uci_all_runs.csv`

## Reproducibility Settings

The benchmark defaults are defined in `uci_benchmark_algorithms.py`:

- `DEFAULT_RANDOM_STATE = 42`
- `DEFAULT_MAX_ITER = 5000`
- `DEFAULT_TOL = 1e-5`
- `n_init = 5` in `run_benchmark.py`

The five initialization seeds are deterministic:

```text
42, 43, 44, 45, 46
```

Runtime columns can vary by machine because they measure wall-clock time.

## Files

- `run_benchmark.py`: one-command benchmark runner.
- `uci_benchmark_algorithms.py`: algorithm implementations, dataset loaders, metrics, and benchmark logic.
- `DA_FCM_UCI_Benchmark.ipynb`: notebook interface using the same benchmark code.
- `uci_data/`: local copies of the UCI datasets, so the benchmark can run without downloading.
- `requirements.txt`: pinned Python package versions.
- `DATA_SOURCES.md`: source URLs for the bundled UCI datasets.

## Custom Runs

You can change the output directory:

```bash
python run_benchmark.py --results-dir my_results
```

You can run with a different iteration cap:

```bash
python run_benchmark.py --max-iter 1000
```
