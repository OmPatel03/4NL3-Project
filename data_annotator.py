from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

ALLOWED_GENRES = [
    "Drama",
    "Comedy",
    "Action",
    "Thriller / Crime",
    "Romance",
    "Horror",
    "Science Fiction",
    "Fantasy",
]


def load_json_array(path: Path) -> list[Any]:
    try:
        raw_json = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Failed to read input file '{path}': {exc}") from exc

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Input file '{path}' is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})."
        ) from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Input JSON must be a top-level array, got {type(data).__name__}."
        )

    return data


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def extract_valid_movies(rows: list[Any]) -> tuple[list[dict[str, str]], list[int]]:
    valid_movies: list[dict[str, str]] = []
    skipped_records: list[int] = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            skipped_records.append(index)
            continue
        
        id = row.get("id")
        title = row.get("title")
        overview = row.get("overview")
        if not (_is_non_empty_string(title) and _is_non_empty_string(overview)):
            skipped_records.append(index)
            continue

        valid_movies.append({"id": id, "title": title, "overview": overview})

    return valid_movies, skipped_records


def load_existing_annotations(path: Path) -> list[Any]:
    if not path.exists():
        raise ValueError(
            f"Resume requested but output file '{path}' does not exist. "
            "Use the same partial output file or restart with -start-at 1."
        )

    try:
        raw_json = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Failed to read output file '{path}': {exc}") from exc

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Output file '{path}' is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})."
        ) from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Resume output must be a top-level array, got {type(data).__name__}."
        )

    return data


def _preview_text(value: Any, limit: int = 60) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def validate_resume_prefix(
    existing: list[Any], valid_movies: list[dict[str, str]], start_at: int
) -> list[dict[str, str]]:
    expected_len = start_at - 1
    actual_len = len(existing)
    if actual_len != expected_len:
        raise ValueError(
            "Resume output length mismatch: "
            f"expected {expected_len} row(s) for -start-at {start_at}, "
            f"found {actual_len}."
        )

    prefix: list[dict[str, str]] = []
    for idx in range(expected_len):
        row = existing[idx]
        display_idx = idx + 1
        if not isinstance(row, dict):
            raise ValueError(
                f"Resume output schema mismatch at row {display_idx}: "
                f"expected object, found {type(row).__name__}."
            )

        expected_title = valid_movies[idx]["title"]
        expected_synopsis = valid_movies[idx]["overview"]
        actual_title = row.get("title")
        actual_synopsis = row.get("synopsis")
        actual_manual_genre = row.get("manual_genre")

        if actual_title != expected_title:
            raise ValueError(
                f"Resume output mismatch at row {display_idx}: "
                f"expected title '{_preview_text(expected_title)}', "
                f"found '{_preview_text(actual_title)}'."
            )

        if actual_synopsis != expected_synopsis:
            raise ValueError(
                f"Resume output mismatch at row {display_idx}: "
                "synopsis does not match current input overview "
                f"(found '{_preview_text(actual_synopsis)}')."
            )

        if (
            not _is_non_empty_string(actual_manual_genre)
            or actual_manual_genre not in ALLOWED_GENRES
        ):
            raise ValueError(
                f"Resume output schema mismatch at row {display_idx}: "
                "manual_genre must be one of "
                f"{ALLOWED_GENRES}, found {_preview_text(actual_manual_genre)!r}."
            )

        prefix.append(
            {
                "title": actual_title,
                "synopsis": actual_synopsis,
                "manual_genre": actual_manual_genre,
            }
        )

    return prefix


def build_resume_command(
    input_json: Path, output_json: Path, start_at: int, fast: bool
) -> str:
    command_parts = ["python", "data_annotator.py"]
    if fast:
        command_parts.append("-fast")
    command_parts.extend(
        [
            "-start-at",
            str(start_at),
            str(input_json),
            str(output_json),
        ]
    )
    return " ".join(shlex.quote(part) for part in command_parts)


def _read_single_key() -> str:
    if os.name == "nt":
        import msvcrt

        return msvcrt.getwch()

    import termios
    import tty

    fd = sys.stdin.fileno()
    original_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)


def prompt_for_genre(
    title: str, overview: str, idx: int, total: int, single_key_mode: bool
) -> str:
    percent = int((idx / total) * 100) if total else 100

    print()
    print(f"[{idx}/{total} | {percent}%]")
    print(f"Title: {title}")
    print("Synopsis:")
    print(overview)
    print("Genre options:")
    for option_num, genre in enumerate(ALLOWED_GENRES, start=1):
        print(f"  {option_num}. {genre}")

    while True:
        if single_key_mode:
            prompt_text = f"Select genre [1-{len(ALLOWED_GENRES)}]: "
            print(prompt_text, end="", flush=True)
            try:
                choice = _read_single_key()
            except (OSError, ValueError):
                print()
                print("Single-key mode unavailable. Falling back to Enter-based input.")
                single_key_mode = False
                continue

            if choice == "\x03":
                raise KeyboardInterrupt
            if choice in ("\r", "\n"):
                print()
                continue

            print(choice)
            choice = choice.strip()
        else:
            choice = input(f"Select genre [1-{len(ALLOWED_GENRES)}]: ").strip()

        if not choice.isdigit():
            print("Invalid input. Please enter a number.")
            continue

        choice_index = int(choice)
        if not 1 <= choice_index <= len(ALLOWED_GENRES):
            print(f"Invalid choice. Please enter a number from 1 to {len(ALLOWED_GENRES)}.")
            continue

        return ALLOWED_GENRES[choice_index - 1]


def annotate_movies(
    valid_movies: list[dict[str, str]],
    start_at: int,
    annotations_prefix: list[dict[str, str]],
    single_key_mode: bool,
) -> list[dict[str, str]]:
    annotations = annotations_prefix
    total = len(valid_movies)

    for idx in range(start_at, total + 1):
        movie = valid_movies[idx - 1]
        selected_genre = prompt_for_genre(
            movie["title"],
            movie["overview"],
            idx=idx,
            total=total,
            single_key_mode=single_key_mode,
        )
        annotations.append(
            {
                "id": movie["id"],
                "title": movie["title"],
                "synopsis": movie["overview"],
                "manual_genre": selected_genre,
            }
        )

    return annotations


def write_output(path: Path, annotations: list[dict[str, str]]) -> None:
    try:
        serialized = json.dumps(annotations, indent=2, ensure_ascii=False)
        path.write_text(f"{serialized}\n", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Failed to write output file '{path}': {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Manually annotate movie genres from an input JSON array and write "
            "a simplified output JSON array."
        )
    )
    parser.add_argument("input_json", type=Path, help="Path to input movie JSON array.")
    parser.add_argument(
        "output_json",
        type=Path,
        help="Path to output annotation JSON array (will be overwritten).",
    )
    parser.add_argument(
        "-fast",
        action="store_true",
        help="Enable single-key genre selection mode (no Enter key required).",
    )
    parser.add_argument(
        "-start-at",
        type=int,
        default=1,
        help="1-based valid-item index to resume annotation from.",
    )
    args = parser.parse_args(argv)

    if args.start_at < 1:
        print("Error: -start-at must be >= 1.", file=sys.stderr)
        return 1

    try:
        rows = load_json_array(args.input_json)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    valid_movies, skipped_records = extract_valid_movies(rows)

    print(f"Loaded {len(rows)} records from: {args.input_json}")
    if skipped_records:
        print(f"Skipping {len(skipped_records)} invalid record(s):")
        for index in skipped_records:
            print(f"  - index {index}: missing/invalid title or overview")
    else:
        print("No invalid records found.")

    if not valid_movies:
        print("No valid movies to annotate. Writing empty output array.")

    total_valid = len(valid_movies)
    max_start_at = total_valid + 1
    if args.start_at > max_start_at:
        print(
            "Error: invalid -start-at range. "
            f"Expected 1..{max_start_at} based on {total_valid} valid movie(s), "
            f"got {args.start_at}.",
            file=sys.stderr,
        )
        return 1

    annotations_prefix: list[dict[str, str]] = []
    if args.start_at > 1:
        try:
            existing_output = load_existing_annotations(args.output_json)
            annotations_prefix = validate_resume_prefix(
                existing_output, valid_movies, start_at=args.start_at
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print(
            f"Resuming annotation from valid-item index {args.start_at} "
            f"(loaded {len(annotations_prefix)} existing row(s))."
        )

    single_key_mode = False
    if args.fast:
        if sys.stdin.isatty() and sys.stdout.isatty():
            single_key_mode = True
            print("Single-key mode enabled: press 1-9 to select (no Enter needed).")
        else:
            print(
                "Warning: -fast requires an interactive terminal. "
                "Falling back to Enter-based input."
            )

    annotations = list(annotations_prefix)
    try:
        annotations = annotate_movies(
            valid_movies,
            start_at=args.start_at,
            annotations_prefix=annotations,
            single_key_mode=single_key_mode,
        )
    except KeyboardInterrupt:
        print("\nAnnotation interrupted. Saving partial progress...")
        try:
            write_output(args.output_json, annotations)
        except ValueError as exc:
            print(f"Error: unable to save partial output: {exc}", file=sys.stderr)
            return 1

        next_index = len(annotations) + 1
        resume_command = build_resume_command(
            args.input_json,
            args.output_json,
            start_at=next_index,
            fast=args.fast,
        )
        print(f"Partial progress saved to: {args.output_json}")
        print(f"Resume with -start-at {next_index}")
        print(f"Command: {resume_command}")
        return 130

    try:
        write_output(args.output_json, annotations)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print()
    print("Annotation complete.")
    if args.start_at > 1:
        print(f"Resumed from valid-item index: {args.start_at}")
    print(f"Total input records: {len(rows)}")
    print(f"Skipped invalid records: {len(skipped_records)}")
    print(f"Successfully annotated: {len(annotations)}")
    print(f"Output written to: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
