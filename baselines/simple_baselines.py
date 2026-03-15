"""
Simple Baselines for Movie Genre Classification
=================================================
Two rule-based baselines that require no training:

1. **Random Baseline**  – assigns a label uniformly at random from the
   set of labels seen in the training split.
2. **Majority Baseline** – always predicts the most frequent label in
   the training split.

Usage
-----
    python simple_baselines.py --train <train.json> --eval <eval.json> --output <predictions.json>

By default both baselines are run and results are printed.  The output
file stores the predictions of whichever baseline scored higher on the
eval set (as required by the project spec).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_json_array(path: Path) -> list[dict[str, Any]]:
    """Load a JSON file that contains a top-level array of objects."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array in '{path}'.")
    return data


def accuracy(predictions: list[str], gold: list[str]) -> float:
    """Compute simple accuracy."""
    assert len(predictions) == len(gold)
    correct = sum(p == g for p, g in zip(predictions, gold))
    return correct / len(gold) if gold else 0.0


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def random_baseline(labels: list[str], n: int, seed: int = 42) -> list[str]:
    """Return *n* predictions sampled uniformly from *labels*."""
    rng = random.Random(seed)
    return [rng.choice(labels) for _ in range(n)]


def majority_baseline(majority_label: str, n: int) -> list[str]:
    """Return *n* copies of the majority label."""
    return [majority_label] * n


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Random & Majority baselines for movie-genre classification."
    )
    parser.add_argument("--train", type=Path, required=True,
                        help="Path to the training JSON (ground_truth split).")
    parser.add_argument("--eval", type=Path, required=True,
                        help="Path to the evaluation JSON (val or test split).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write the winning baseline's predictions.")
    args = parser.parse_args(argv)

    train_rows = load_json_array(args.train)
    eval_rows = load_json_array(args.eval)

    # --- label statistics from training data ---
    train_labels = [row["manual_genre"] for row in train_rows]
    label_counts = Counter(train_labels)
    labels_list = sorted(label_counts)           # unique labels
    majority_label = label_counts.most_common(1)[0][0]

    print(f"Training set size : {len(train_rows)}")
    print(f"Eval set size     : {len(eval_rows)}")
    print(f"Labels            : {labels_list}")
    print(f"Majority label    : {majority_label} "
          f"({label_counts[majority_label]}/{len(train_rows)} = "
          f"{label_counts[majority_label] / len(train_rows):.2%})")
    print()

    # --- ground truth for the eval split ---
    gold = [row["manual_genre"] for row in eval_rows]

    # --- random baseline ---
    rand_preds = random_baseline(labels_list, len(eval_rows))
    rand_acc = accuracy(rand_preds, gold)
    print(f"Random Baseline   accuracy: {rand_acc:.4f}")

    # --- majority baseline ---
    maj_preds = majority_baseline(majority_label, len(eval_rows))
    maj_acc = accuracy(maj_preds, gold)
    print(f"Majority Baseline accuracy: {maj_acc:.4f}")

    # --- pick the best simple baseline ---
    if maj_acc >= rand_acc:
        best_name, best_preds = "Majority", maj_preds
    else:
        best_name, best_preds = "Random", rand_preds

    print(f"\nBest simple baseline: {best_name} ({max(rand_acc, maj_acc):.4f})")

    # --- write predictions in the same format as the Naive Bayes starter ---
    predictions = [
        {"title": row["title"], "manual_genre": pred}
        for row, pred in zip(eval_rows, best_preds)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Predictions written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
