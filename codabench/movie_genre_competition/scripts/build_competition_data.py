from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = ROOT.parents[1] / "Final Annotate" / "ground_truth.json"

LABELS = [
    "Drama",
    "Comedy",
    "Action",
    "Thriller / Crime",
    "Romance",
    "Horror",
    "Science Fiction",
    "Fantasy",
]

TRAIN_END = 701
VAL_END = 851


def read_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in '{path}'.")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Row {index} in '{path}' is not a JSON object.")
        rows.append(item)
    return rows


def strip_labels(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"title": str(row["title"]), "synopsis": str(row["synopsis"])} for row in rows]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    rows = read_rows(GROUND_TRUTH)
    if len(rows) != 1001:
        raise ValueError(f"Expected 1001 rows in ground_truth.json, found {len(rows)}.")

    train_rows = rows[:TRAIN_END]
    val_rows = rows[TRAIN_END:VAL_END]
    test_rows = rows[VAL_END:]

    if train_rows[0]["title"] != "Inception" or train_rows[-1]["title"] != "Spy":
        raise ValueError("Training split boundaries do not match Inception -> Spy.")
    if val_rows[0]["title"] != "Lilo & Stitch" or val_rows[-1]["title"] != "American Gangster":
        raise ValueError("Validation split boundaries do not match Lilo & Stitch -> American Gangster.")
    if test_rows[0]["title"] != "Hook" or test_rows[-1]["title"] != "The Giver":
        raise ValueError("Test split boundaries do not match Hook -> The Giver.")

    public_data = ROOT / "public_data"
    write_json(public_data / "train.json", train_rows)
    write_json(public_data / "validation_input.json", strip_labels(val_rows))
    write_json(public_data / "label_names.json", LABELS)

    dev_input = ROOT / "tasks" / "development" / "input_data"
    dev_ref = ROOT / "tasks" / "development" / "reference_data"
    final_input = ROOT / "tasks" / "final" / "input_data"
    final_ref = ROOT / "tasks" / "final" / "reference_data"

    write_json(dev_input / "train.json", train_rows)
    write_json(dev_input / "eval.json", strip_labels(val_rows))
    write_json(dev_input / "label_names.json", LABELS)
    write_json(dev_ref / "labels.json", val_rows)
    write_json(
        dev_ref / "phase_config.json",
        {
            "phase": "development",
            "leaderboard_macro_f1_key": "dev_public_macro_f1",
            "leaderboard_accuracy_key": "dev_public_accuracy",
        },
    )

    write_json(final_input / "train.json", train_rows)
    write_json(final_input / "eval.json", strip_labels(test_rows))
    write_json(final_input / "label_names.json", LABELS)
    write_json(final_ref / "labels.json", test_rows)
    write_json(
        final_ref / "phase_config.json",
        {
            "phase": "final",
            "leaderboard_macro_f1_key": "final_private_macro_f1",
            "leaderboard_accuracy_key": "final_private_accuracy",
        },
    )

    print("Created Codabench data artifacts.")
    print(f"Train rows: {len(train_rows)}")
    print(f"Validation rows: {len(val_rows)}")
    print(f"Test rows: {len(test_rows)}")


if __name__ == "__main__":
    main()
