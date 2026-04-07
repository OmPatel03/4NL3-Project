# Public Data Schema

## train.json

Labeled training rows:

```json
{
  "title": "Movie title",
  "synopsis": "Movie synopsis",
  "manual_genre": "Canonical genre label"
}
```

## validation_input.json

Unlabeled validation inputs:

```json
{
  "title": "Movie title",
  "synopsis": "Movie synopsis"
}
```

## label_names.json

JSON array of canonical label names used by the scorer.
