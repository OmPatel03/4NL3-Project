# Submission

## Submission type

This competition accepts **code submissions**.

## Required file

Each submission zip must contain a top-level `run.py` file.

## Runtime contract

During the ingestion step, Codabench calls:

```bash
python3 run.py --train /app/input_data/train.json --eval /app/input_data/eval.json --output /app/output/predictions.json
```

## Expected output

Your `run.py` must write `predictions.json` as a JSON array in the same order as `eval.json`.

Each output row must contain:

```json
{
  "title": "Movie title",
  "manual_genre": "Predicted label"
}
```

## Allowed labels

Predicted labels must be exactly one of:

- Drama
- Comedy
- Action
- Thriller / Crime
- Romance
- Horror
- Science Fiction
- Fantasy

## Starter kit

A minimal runnable starter kit is included with this bundle. It uses only the Python standard library so it works with the default Codabench runtime image.
