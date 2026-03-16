"""Majority Baseline - predicts the most frequent genre in training."""

import json
from collections import Counter
from pathlib import Path

# Load ground truth
data = json.loads(Path("Final Annotate/ground_truth.json").read_text(encoding="utf-8"))

# Split: train (0-700), val (701-850), test (851-1000)
train = data[0:701]
val = data[701:851]
test = data[851:1001]

# Find majority label
train_labels = [row["manual_genre"] for row in train]
majority_label = Counter(train_labels).most_common(1)[0][0]

# Evaluate on val
val_gold = [row["manual_genre"] for row in val]
val_preds = [majority_label] * len(val)
val_acc = sum(p == g for p, g in zip(val_preds, val_gold)) / len(val)

# Evaluate on test
test_gold = [row["manual_genre"] for row in test]
test_preds = [majority_label] * len(test)
test_acc = sum(p == g for p, g in zip(test_preds, test_gold)) / len(test)

print(f"Majority label: {majority_label}")
print(f"Val accuracy  : {val_acc:.4f}")
print(f"Test accuracy : {test_acc:.4f}")
