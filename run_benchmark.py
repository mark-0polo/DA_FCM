from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from uci_benchmark_algorithms import (
    DEFAULT_MAX_ITER,
    DEFAULT_N_SEEDS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TOL,
    STATISTICAL_TEST_ALPHA,
    benchmark_all_methods,
    load_uci_datasets,
)


METHODS = ["FCM", "PCM", "PFCM", "DPC-FCM", "eFCM", "DA-FCM"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reproducible DA-FCM UCI benchmark."
    )
    parser.add_argument("--data-dir", default="uci_data", help="Directory containing UCI data files.")
    parser.add_argument("--results-dir", default="results", help="Directory where output CSVs are written.")
    parser.add_argument("--n-init", type=int, default=DEFAULT_N_SEEDS, help="Random seeds per dataset-method pair.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Base random seed.")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help="Maximum iterations per fit.")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help="Convergence tolerance.")
    parser.add_argument("--alpha", type=float, default=STATISTICAL_TEST_ALPHA, help="Significance level for Friedman/Nemenyi tests.")
    return parser.parse_args()


def save_outputs(
    summary_df: pd.DataFrame,
    all_runs_df: pd.DataFrame,
    friedman_df: pd.DataFrame,
    rank_df: pd.DataFrame,
    nemenyi_df: pd.DataFrame,
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(results_dir / "all_algorithms_uci_benchmark_summary.csv", index=False)
    all_runs_df.to_csv(results_dir / "all_algorithms_uci_all_runs.csv", index=False)
    friedman_df.to_csv(results_dir / "all_algorithms_uci_friedman_tests.csv", index=False)
    rank_df.to_csv(results_dir / "all_algorithms_uci_average_ranks.csv", index=False)
    nemenyi_df.to_csv(results_dir / "all_algorithms_uci_nemenyi_tests.csv", index=False)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    data_dir = (root / args.data_dir).resolve()
    results_dir = (root / args.results_dir).resolve()

    print("DA-FCM UCI benchmark")
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Methods: {', '.join(METHODS)}")
    print(
        "Settings: "
        f"n_init={args.n_init}, random_state={args.random_state}, "
        f"max_iter={args.max_iter}, tol={args.tol}, alpha={args.alpha}"
    )
    print(f"Seeds: {args.random_state} to {args.random_state + args.n_init - 1}")

    datasets = load_uci_datasets(data_dir)
    print("Datasets:")
    for dataset in datasets:
        print(
            f"  - {dataset['name']}: "
            f"{dataset['samples']} rows, {dataset['features']} features, "
            f"{dataset['classes']} classes"
        )

    summary_df, all_runs_df, friedman_df, rank_df, nemenyi_df = benchmark_all_methods(
        datasets,
        methods=METHODS,
        n_init=args.n_init,
        random_state=args.random_state,
        max_iter=args.max_iter,
        tol=args.tol,
        alpha=args.alpha,
    )

    save_outputs(summary_df, all_runs_df, friedman_df, rank_df, nemenyi_df, results_dir)

    print("Saved:")
    print(f"  {results_dir / 'all_algorithms_uci_benchmark_summary.csv'}")
    print(f"  {results_dir / 'all_algorithms_uci_all_runs.csv'}")
    print(f"  {results_dir / 'all_algorithms_uci_friedman_tests.csv'}")
    print(f"  {results_dir / 'all_algorithms_uci_average_ranks.csv'}")
    print(f"  {results_dir / 'all_algorithms_uci_nemenyi_tests.csv'}")


if __name__ == "__main__":
    main()
