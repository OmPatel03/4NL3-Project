# Evaluation

## Primary metric

The competition is ranked by **macro F1**.

Macro F1 gives equal weight to each genre and is therefore more appropriate than raw accuracy for an imbalanced 8-class label set.

## Secondary metric

The scoring program also computes accuracy as a diagnostic metric, but it is not shown as a leaderboard column.

## Validation rules

A submission is valid only if:

- `predictions.json` is a JSON array
- the row count matches the evaluation split exactly
- the prediction order matches the provided `eval.json`
- each prediction object contains `title` and `manual_genre`
- every `manual_genre` value matches one of the canonical label names

Invalid submissions fail scoring rather than receiving a misleading numeric score.
