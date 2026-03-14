from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


class NaiveBayesClassifier:
    def __init__(self) -> None:
        self.label_counts: Counter[str] = Counter()
        self.token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.total_tokens: Counter[str] = Counter()
        self.vocab: set[str] = set()
        self.labels: list[str] = []

    def fit(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            label = str(row["manual_genre"])
            self.label_counts[label] += 1
            tokens = tokenize(build_text(row))
            self.token_counts[label].update(tokens)
            self.total_tokens[label] += len(tokens)
            self.vocab.update(tokens)
        self.labels = sorted(self.label_counts)

    def predict_one(self, row: dict[str, Any]) -> str:
        tokens = tokenize(build_text(row))
        total_docs = sum(self.label_counts.values())
        vocab_size = max(len(self.vocab), 1)
        best_label = ""
        best_score = float("-inf")

        for label in self.labels:
            prior = math.log(self.label_counts[label] / total_docs)
            score = prior
            denom = self.total_tokens[label] + vocab_size
            token_counter = self.token_counts[label]
            for token in tokens:
                score += math.log((token_counter[token] + 1) / denom)
            if score > best_score:
                best_score = score
                best_label = label

        return best_label or self.labels[0]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_text(row: dict[str, Any]) -> str:
    return f"{row.get('title', '')} {row.get('synopsis', '')}"


def load_json_array(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array in '{path}'.")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Row {index} in '{path}' is not a JSON object.")
        rows.append(item)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    train_rows = load_json_array(args.train)
    eval_rows = load_json_array(args.eval)

    model = NaiveBayesClassifier()
    model.fit(train_rows)

    predictions = [
        {"title": row["title"], "manual_genre": model.predict_one(row)}
        for row in eval_rows
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
