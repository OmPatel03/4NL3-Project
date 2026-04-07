import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score

LABEL_NAMES = [
    "Drama",
    "Comedy",
    "Action",
    "Thriller / Crime",
    "Romance",
    "Horror",
    "Science Fiction",
    "Fantasy",
]


def load_json_array(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array in '{path}'.")
    return data


def get_labels(rows):
    return [row["manual_genre"] for row in rows]


def get_prediction_labels(rows):
    return [row["manual_genre"] for row in rows]


def save_confusion_matrix(gold, predictions, label_names, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(gold, predictions, labels=label_names),
        display_labels=label_names,
    ).plot(ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_error_csv(rows, gold, predictions, output_path):
    error_rows = []
    for row, actual, pred in zip(rows, gold, predictions):
        if actual != pred:
            error_rows.append(
                {
                    "title": row["title"],
                    "gold": actual,
                    "predicted": pred,
                    "synopsis_preview": row["synopsis"][:240],
                }
            )
    pd.DataFrame(error_rows[:10]).to_csv(output_path, index=False)


def run_model(script_path, train_path, eval_path, output_path):
    command = [
        sys.executable,
        str(script_path),
        "--train",
        str(train_path),
        "--eval",
        str(eval_path),
        "--output",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def check_predictions(prediction_rows, source_rows, label_names):
    if len(prediction_rows) != len(source_rows):
        raise ValueError("Prediction count does not match input count.")

    for pred_row, source_row in zip(prediction_rows, source_rows):
        if pred_row["title"] != source_row["title"]:
            raise ValueError("Prediction titles do not match source rows.")
        if pred_row["manual_genre"] not in label_names:
            raise ValueError(f"Invalid label found: {pred_row['manual_genre']}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    final_dir = repo_root / "final"
    data_dir = final_dir / "data"
    output_dir = final_dir / "outputs"
    figure_dir = final_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.json"
    val_path = data_dir / "val.json"
    public_train_path = data_dir / "public_train.json"
    public_eval_path = data_dir / "public_eval.json"

    val_rows = load_json_array(val_path)
    val_labels = get_labels(val_rows)
    public_eval_rows = load_json_array(public_eval_path)

    validation_jobs = [
        ("Majority Baseline", "baseline", final_dir / "majority_baseline.py", output_dir / "majority_val_predictions.json"),
        ("Logistic Regression Baseline", "baseline", final_dir / "logistic_regression_baseline.py", output_dir / "logistic_regression_val_predictions.json"),
        ("ComplementNB + Count Bigrams", "required_model", final_dir / "complement_nb_model.py", output_dir / "complement_nb_val_predictions.json"),
        ("RidgeClassifier + Word TF-IDF", "required_model", final_dir / "ridge_tfidf_model.py", output_dir / "ridge_tfidf_val_predictions.json"),
        ("LinearSVC + Word/Char TF-IDF", "required_model", final_dir / "linear_svc_model.py", output_dir / "linear_svc_val_predictions.json"),
        ("Zero-shot NLI (DistilBART MNLI)", "required_model", final_dir / "zero_shot_nli_model.py", output_dir / "zero_shot_val_predictions.json"),
    ]

    results = []
    saved_predictions = {}

    for model_name, model_group, script_path, prediction_path in validation_jobs:
        run_model(script_path, train_path, val_path, prediction_path)
        prediction_rows = load_json_array(prediction_path)
        check_predictions(prediction_rows, val_rows, LABEL_NAMES)
        predictions = get_prediction_labels(prediction_rows)
        saved_predictions[model_name] = predictions
        results.append(
            {
                "model_name": model_name,
                "model_group": model_group,
                "macro_f1": f1_score(val_labels, predictions, average="macro"),
                "accuracy": accuracy_score(val_labels, predictions),
            }
        )

    comparison_df = pd.DataFrame(results).sort_values(["macro_f1", "accuracy"], ascending=False)
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    print(comparison_df.to_string(index=False))

    majority_f1 = comparison_df[comparison_df["model_name"] == "Majority Baseline"]["macro_f1"].iloc[0]
    for row in comparison_df.itertuples():
        if row.model_group == "required_model":
            print(f"{row.model_name} beats majority on macro F1: {row.macro_f1 > majority_f1}")

    best_supervised_name = ""
    best_supervised_f1 = -1
    for row in results:
        if row["model_group"] == "required_model" and "Zero-shot" not in row["model_name"]:
            if row["macro_f1"] > best_supervised_f1:
                best_supervised_name = row["model_name"]
                best_supervised_f1 = row["macro_f1"]

    save_confusion_matrix(
        val_labels,
        saved_predictions[best_supervised_name],
        LABEL_NAMES,
        best_supervised_name,
        figure_dir / "best_supervised_confusion_matrix.png",
    )
    save_confusion_matrix(
        val_labels,
        saved_predictions["Zero-shot NLI (DistilBART MNLI)"],
        LABEL_NAMES,
        "Zero-shot NLI (DistilBART MNLI)",
        figure_dir / "zero_shot_confusion_matrix.png",
    )

    save_error_csv(
        val_rows,
        val_labels,
        saved_predictions[best_supervised_name],
        output_dir / "best_supervised_errors.csv",
    )
    save_error_csv(
        val_rows,
        val_labels,
        saved_predictions["Zero-shot NLI (DistilBART MNLI)"],
        output_dir / "zero_shot_errors.csv",
    )

    public_jobs = [
        (final_dir / "complement_nb_model.py", output_dir / "complementnb_count_bigrams_public_validation_predictions.json"),
        (final_dir / "ridge_tfidf_model.py", output_dir / "ridgeclassifier_word_tf_idf_public_validation_predictions.json"),
        (final_dir / "linear_svc_model.py", output_dir / "linearsvc_word_char_tf_idf_public_validation_predictions.json"),
        (final_dir / "zero_shot_nli_model.py", output_dir / "zero_shot_nli_distilbart_mnli_public_validation_predictions.json"),
    ]

    for script_path, prediction_path in public_jobs:
        run_model(script_path, public_train_path, public_eval_path, prediction_path)
        prediction_rows = load_json_array(prediction_path)
        check_predictions(prediction_rows, public_eval_rows, LABEL_NAMES)

    print("\nSaved comparison table, figures, error CSVs, and public prediction files.")


if __name__ == "__main__":
    main()
