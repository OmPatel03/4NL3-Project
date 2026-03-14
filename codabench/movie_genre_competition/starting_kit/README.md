# Starter Kit

This starter kit contains a single-file baseline submission named `run.py`.

## What it does

- reads `train.json`
- trains a simple multinomial naive Bayes text classifier using only the Python standard library
- predicts one label for each row in `eval.json`
- writes `predictions.json` in Codabench's required format

## Local usage

```bash
python3 run.py --train train.json --eval eval.json --output predictions.json
```

## Submission packaging

Zip the contents of this folder so that `run.py` is at the top level of the submission archive.
