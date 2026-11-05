# split_synthetic.py
"""Utility to split the full synthetic dataset into a 70/30 train‑test split.
The training split is written to the normal data folder (used by the pipeline
when ``--source synthetic`` is selected). The test split is written to the
``testing_data`` folder, which the ``run_test_data.py`` script already points
to.
"""

import argparse
import pathlib
import pandas as pd
import numpy as np

def split_synthetic(csv_path: pathlib.Path, train_frac: float = 0.7, seed: int = 42):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Write training set (used by normal pipeline)
    train_path = pathlib.Path("data") / "synthetic" / "synthetic_transactions.csv"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)

    # Write test set (used by the test harness)
    test_path = pathlib.Path("testing_data") / "synthetic" / "synthetic_transactions.csv"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(test_path, index=False)

    print(f"Train rows: {len(train_df)} -> {train_path}")
    print(f"Test rows: {len(test_df)} -> {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split the full synthetic CSV into train/test.")
    parser.add_argument("--path", required=True, help="Path to the full synthetic CSV (e.g., data/synthetic/synthetic_transactions.csv)")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Fraction of rows for training (default 0.7)")
    args = parser.parse_args()
    split_synthetic(pathlib.Path(args.path), args.train_frac)
