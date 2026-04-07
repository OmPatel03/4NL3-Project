from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_json_array(path: Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in '{path}'.")
    return data


def validate_predictions(eval_rows: list[Any], pred_rows: list[Any]) -> None:
    if len(eval_rows) != len(pred_rows):
        raise ValueError(
            f"Prediction row count mismatch: expected {len(eval_rows)}, got {len(pred_rows)}."
        )

    for index, (eval_row, pred_row) in enumerate(zip(eval_rows, pred_rows), start=1):
        if not isinstance(pred_row, dict):
            raise ValueError(f"Prediction row {index} is not a JSON object.")
        if pred_row.get("title") != eval_row.get("title"):
            raise ValueError(
                f"Title mismatch at row {index}: expected {eval_row.get('title')!r}, "
                f"got {pred_row.get('title')!r}."
            )
        if "manual_genre" not in pred_row:
            raise ValueError(f"Prediction row {index} is missing 'manual_genre'.")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 4:
        print(
            "Usage: ingestion.py <input_data_dir> <output_dir> <program_dir> <submission_dir>",
            file=sys.stderr,
        )
        return 1

    input_data_dir = Path(args[0]).resolve()
    output_dir = Path(args[1]).resolve()
    submission_dir = Path(args[3]).resolve()

    train_path = input_data_dir / "train.json"
    eval_path = input_data_dir / "eval.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.json"

    run_script = (submission_dir / "run.py").resolve()
    if not run_script.exists():
        print("Error: submission zip must contain a top-level run.py", file=sys.stderr)
        return 1

    command = [
        sys.executable,
        str(run_script),
        "--train",
        str(train_path),
        "--eval",
        str(eval_path),
        "--output",
        str(predictions_path),
    ]

    completed = subprocess.run(command, cwd=submission_dir, check=False)
    if completed.returncode != 0:
        print(f"Error: participant program exited with status {completed.returncode}.", file=sys.stderr)
        return completed.returncode

    if not predictions_path.exists():
        print("Error: participant program did not create predictions.json.", file=sys.stderr)
        return 1

    try:
        eval_rows = load_json_array(eval_path)
        pred_rows = load_json_array(predictions_path)
        validate_predictions(eval_rows, pred_rows)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Error: invalid prediction output: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
