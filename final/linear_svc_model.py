import argparse
import json
from pathlib import Path

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC


def load_json_array(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array in '{path}'.")
    return data


def build_text(row):
    return f"{row.get('title', '')} {row.get('synopsis', '')}"


def main():
    parser = argparse.ArgumentParser(description="Linear SVC with word and character TF-IDF features.")
    parser.add_argument("--train", type=Path, required=True, help="Path to the training JSON.")
    parser.add_argument("--eval", type=Path, required=True, help="Path to the evaluation JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write predictions.")
    args = parser.parse_args()

    train_rows = load_json_array(args.train)
    eval_rows = load_json_array(args.eval)

    train_texts = [build_text(row) for row in train_rows]
    eval_texts = [build_text(row) for row in eval_rows]
    train_labels = [row["manual_genre"] for row in train_rows]
    eval_labels = [row["manual_genre"] for row in eval_rows if "manual_genre" in row]

    word_vectorizer = TfidfVectorizer(
        token_pattern=r"[A-Za-z0-9']+",
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=30_000,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        ngram_range=(3, 5),
        sublinear_tf=True,
        max_features=30_000,
    )

    x_train_word = word_vectorizer.fit_transform(train_texts)
    x_eval_word = word_vectorizer.transform(eval_texts)
    x_train_char = char_vectorizer.fit_transform(train_texts)
    x_eval_char = char_vectorizer.transform(eval_texts)

    x_train = hstack([x_train_word, x_train_char])
    x_eval = hstack([x_eval_word, x_eval_char])

    clf = LinearSVC(C=0.8, class_weight="balanced")
    clf.fit(x_train, train_labels)
    predictions = clf.predict(x_eval).tolist()

    print(f"Training set : {len(train_rows)} rows")
    print(f"Eval set     : {len(eval_rows)} rows")
    print(f"Word vocab   : {len(word_vectorizer.vocabulary_)}")
    print(f"Char vocab   : {len(char_vectorizer.vocabulary_)}")

    if len(eval_labels) == len(eval_rows):
        acc = accuracy_score(eval_labels, predictions)
        macro_f1 = f1_score(eval_labels, predictions, average="macro")
        print(f"\nEval accuracy: {acc:.4f}")
        print(f"Macro F1     : {macro_f1:.4f}\n")
        print(classification_report(eval_labels, predictions, zero_division=0))
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
