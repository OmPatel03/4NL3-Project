from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ALLOWED_LABELS = [
    "Drama",
    "Comedy",
    "Action",
    "Thriller / Crime",
    "Romance",
    "Horror",
    "Science Fiction",
    "Fantasy",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_array(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in '{path}'.")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Row {index} in '{path}' is not a JSON object.")
        rows.append(item)
    return rows


def validate_predictions(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> None:
    if len(predictions) != len(references):
        raise ValueError(
            f"Prediction row count mismatch: expected {len(references)}, got {len(predictions)}."
        )

    allowed = set(ALLOWED_LABELS)
    for index, (pred_row, ref_row) in enumerate(zip(predictions, references), start=1):
        if pred_row.get("title") != ref_row.get("title"):
            raise ValueError(
                f"Title mismatch at row {index}: expected {ref_row.get('title')!r}, "
                f"got {pred_row.get('title')!r}."
            )
        label = pred_row.get("manual_genre")
        if label not in allowed:
            raise ValueError(
                f"Invalid label at row {index}: {label!r}. Expected one of {ALLOWED_LABELS}."
            )


def accuracy_score(true_labels: list[str], pred_labels: list[str]) -> float:
    if not true_labels:
        return 0.0
    correct = sum(1 for truth, pred in zip(true_labels, pred_labels) if truth == pred)
    return correct / len(true_labels)


def macro_f1_score(true_labels: list[str], pred_labels: list[str]) -> float:
    f1_values: list[float] = []
    for label in ALLOWED_LABELS:
        tp = sum(1 for truth, pred in zip(true_labels, pred_labels) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(true_labels, pred_labels) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(true_labels, pred_labels) if truth == label and pred != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_values.append(f1)
    return sum(f1_values) / len(f1_values)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("Usage: score.py <input_dir> <output_dir>", file=sys.stderr)
        return 1

    input_dir = Path(args[0])
    output_dir = Path(args[1])
    predictions_path = input_dir / "res" / "predictions.json"
    labels_path = input_dir / "ref" / "labels.json"
    phase_config_path = input_dir / "ref" / "phase_config.json"

    try:
        predictions = load_json_array(predictions_path)
        references = load_json_array(labels_path)
        phase_config = load_json(phase_config_path)
        if not isinstance(phase_config, dict):
            raise ValueError("phase_config.json must be a JSON object.")
        validate_predictions(predictions, references)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    true_labels = [str(row["manual_genre"]) for row in references]
    pred_labels = [str(row["manual_genre"]) for row in predictions]
    macro_f1 = macro_f1_score(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)

    leaderboard_macro_key = str(phase_config["leaderboard_macro_f1_key"])
    leaderboard_accuracy_key = str(phase_config["leaderboard_accuracy_key"])
    scores = {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        leaderboard_macro_key: macro_f1,
        leaderboard_accuracy_key: accuracy,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scores.json").write_text(
        json.dumps(scores, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
