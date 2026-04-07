# Data

## Source dataset

The benchmark uses manually adjudicated labels from `Final Annotate/ground_truth.json`.

Each record has the following schema:

```json
{
  "title": "Movie title",
  "synopsis": "Movie synopsis",
  "manual_genre": "One of 8 allowed labels"
}
```

## Public files

Participants can download:

- `train.json`: labeled training set
- `validation_input.json`: unlabeled validation inputs
- `label_names.json`: canonical label list

## Hidden files

The benchmark keeps the following labels hidden from participants:

- validation labels used for public leaderboard scoring
- test labels used for final private scoring

## Label set

The canonical label names are:

- Drama
- Comedy
- Action
- Thriller / Crime
- Romance
- Horror
- Science Fiction
- Fantasy
