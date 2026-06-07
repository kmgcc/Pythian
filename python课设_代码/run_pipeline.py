from __future__ import annotations

import argparse

from src.pipeline import run_full_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the natural light spectrum prediction pipeline.")
    parser.add_argument("--samples", type=int, default=1800, help="Number of simulated samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, result = run_full_pipeline(n_samples=args.samples, seed=args.seed)
    print("Pipeline finished.")
    print(f"Dataset rows: {len(dataset)}")
    print(f"Best model: {result.best_model_name}")
    print(result.metrics.to_string(index=False))


if __name__ == "__main__":
    main()

