from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def load_json_array(path: Path) -> list[dict[str, Any]]:
    """Load a JSON file that contains a top-level array of objects."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array in '{path}'.")
    return data


def build_text(row: dict[str, Any]) -> str:
    """Concatenate `title` and `synopsis` into a single string."""
    return f"{row.get('title', '')} {row.get('synopsis', '')}"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CBOW + Logistic Regression baseline for movie-genre classification."
    )
    parser.add_argument("--train", type=Path, required=True,
                        help="Path to the training JSON.")
    parser.add_argument("--eval", type=Path, required=True,
                        help="Path to the evaluation JSON.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write predictions.")
    parser.add_argument("--use-tfidf", action="store_true", default=False,
                        help="Use TF-IDF weighting instead of raw counts.")
    args = parser.parse_args(argv)

    # --- load data ---
    train_rows = load_json_array(args.train)
    eval_rows = load_json_array(args.eval)

    train_texts = [build_text(r) for r in train_rows]
    eval_texts = [build_text(r) for r in eval_rows]

    train_labels = [r["manual_genre"] for r in train_rows]
    eval_labels = [r["manual_genre"] for r in eval_rows]

    print(f"Training set : {len(train_rows)} rows")
    print(f"Eval set     : {len(eval_rows)} rows")

    # --- label distribution ---
    label_counts = Counter(train_labels)
    print("\nTraining label distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label:20s}: {count:4d}  ({count / len(train_labels):.1%})")

    # --- vectorize ---
    if args.use_tfidf:
        vectorizer = TfidfVectorizer(
            token_pattern=r"[A-Za-z0-9']+",
            lowercase=True,
            max_features=10_000,
        )
        print("\nUsing TF-IDF features")
    else:
        vectorizer = CountVectorizer(
            token_pattern=r"[A-Za-z0-9']+",
            lowercase=True,
            max_features=10_000,
        )
        print("\nUsing raw count (CBOW) features")

    X_train = vectorizer.fit_transform(train_texts)
    X_eval = vectorizer.transform(eval_texts)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Feature matrix : {X_train.shape}")

    # --- train logistic regression ---
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )
    clf.fit(X_train, train_labels)

    # --- evaluate ---
    eval_preds = clf.predict(X_eval).tolist()
    acc = accuracy_score(eval_labels, eval_preds)

    print(f"\nEval accuracy: {acc:.4f}\n")
    print(classification_report(eval_labels, eval_preds, zero_division=0))

    # --- write predictions ---
    predictions = [
        {"title": row["title"], "manual_genre": pred}
        for row, pred in zip(eval_rows, eval_preds)
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
