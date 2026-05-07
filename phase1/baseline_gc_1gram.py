import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

if __package__ is None and __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from phase1.utils import read_manifest


def gc_1gram_features(seq: str) -> np.ndarray:
    seq = seq.upper()
    length = max(len(seq), 1)
    counts = {"A": 0, "C": 0, "G": 0, "T": 0, "N": 0}
    for ch in seq:
        if ch in counts:
            counts[ch] += 1
        else:
            counts["N"] += 1
    gc = (counts["G"] + counts["C"]) / length
    freqs = np.array([
        counts["A"] / length,
        counts["C"] / length,
        counts["G"] / length,
        counts["T"] / length,
        counts["N"] / length,
    ], dtype=np.float32)
    return np.concatenate([[gc], freqs], axis=0)


def gc_only_features(seq: str) -> np.ndarray:
    seq = seq.upper()
    length = max(len(seq), 1)
    gc = (seq.count("G") + seq.count("C")) / length
    return np.array([gc], dtype=np.float32)


def train_baseline(
    features: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    c_grid: List[float],
    max_iter: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    best_val = -1.0
    best_metrics: Dict[str, float] = {}
    best_meta: Dict[str, float] = {}

    for c in c_grid:
        clf = LogisticRegression(
            C=c,
            solver="lbfgs",
            max_iter=max_iter,
            n_jobs=None,
        )
        clf.fit(features[train_mask], labels[train_mask])

        metrics = {}
        for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            probs = clf.predict_proba(features[mask])[:, 1]
            preds = (probs >= 0.5).astype(np.int64)
            metrics[f"{name}_acc"] = accuracy_score(labels[mask], preds)
            metrics[f"{name}_auroc"] = roc_auc_score(labels[mask], probs)

        if metrics["val_auroc"] > best_val:
            best_val = metrics["val_auroc"]
            best_metrics = metrics
            best_meta = {"C": c}

    return best_metrics, best_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="GC + 1-gram baseline for viral vs non-viral.")
    parser.add_argument("--manifest", default="data/phase1/manifest.csv")
    parser.add_argument("--out-dir", default="data/phase1/baselines")
    parser.add_argument("--c-grid", default="0.1,1,10")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--feature", choices=["gc", "gc_1gram"], default="gc_1gram")
    args = parser.parse_args()

    records = read_manifest(args.manifest)
    if args.feature == "gc":
        features = np.stack([gc_only_features(r.sequence) for r in records], axis=0)
    else:
        features = np.stack([gc_1gram_features(r.sequence) for r in records], axis=0)
    labels = np.array([r.label for r in records], dtype=np.int64)
    splits = np.array([r.split for r in records])

    c_grid = [float(x) for x in args.c_grid.split(",")]
    metrics, meta = train_baseline(features, labels, splits, c_grid, args.max_iter)

    os.makedirs(args.out_dir, exist_ok=True)
    out_name = "gc_metrics.csv" if args.feature == "gc" else "gc_1gram_metrics.csv"
    out_path = os.path.join(args.out_dir, out_name)
    row = {"C": meta["C"], **metrics}
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"Wrote baseline metrics to {out_path}")


if __name__ == "__main__":
    main()
