import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score


def load_json_array(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array in '{path}'.")
    return data


def main():
    parser = argparse.ArgumentParser(description="Majority baseline for movie genre classification.")
    parser.add_argument("--train", type=Path, required=True, help="Path to the training JSON.")
    parser.add_argument("--eval", type=Path, required=True, help="Path to the evaluation JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write predictions.")
    args = parser.parse_args()

    train_rows = load_json_array(args.train)
    eval_rows = load_json_array(args.eval)

    train_labels = [row["manual_genre"] for row in train_rows]
    label_counts = Counter(train_labels)
    majority_label = label_counts.most_common(1)[0][0]
    predictions = [majority_label] * len(eval_rows)

    print(f"Training set size : {len(train_rows)}")
    print(f"Eval set size     : {len(eval_rows)}")
    print(f"Majority label    : {majority_label}")

    if eval_rows and all("manual_genre" in row for row in eval_rows):
        gold = [row["manual_genre"] for row in eval_rows]
        acc = accuracy_score(gold, predictions)
        macro_f1 = f1_score(gold, predictions, average="macro")
        print(f"\nEval accuracy: {acc:.4f}")
        print(f"Macro F1     : {macro_f1:.4f}\n")
        print(classification_report(gold, predictions, zero_division=0))
    else:
        print("\nEval labels not present; skipping metric computation.")

    output_rows = []
    for row, pred in zip(eval_rows, predictions):
        output_rows.append({"title": row["title"], "manual_genre": pred})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
