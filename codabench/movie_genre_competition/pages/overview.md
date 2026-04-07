# Overview

This competition asks participants to predict a single dominant movie genre from a movie title and synopsis.

## Task

Each movie must be assigned exactly one label from the following set:

- Drama
- Comedy
- Action
- Thriller / Crime
- Romance
- Horror
- Science Fiction
- Fantasy

This is a single-label text classification benchmark. The intended inputs are the movie `title` and `synopsis`. No other raw metadata is required or expected for participation.

## Competition structure

- Development phase: public validation leaderboard
- Final phase: private test leaderboard
- Submission type: code submission
- Primary metric: macro F1

## Splits

The competition data is derived from the authoritative annotations in `ground_truth.json` using the current row order:

- Training: `Inception` through `Spy` (`701` rows)
- Validation: `Lilo & Stitch` through `American Gangster` (`150` rows)
- Testing: `Hook` through `The Giver` (`150` rows)
