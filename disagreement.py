from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INITIAL = Path("Final Annotate/initial_annotate.json")
DEFAULT_SECOND = Path("Final Annotate/second_pass_annotate.json")
DEFAULT_OUTPUT = Path("Final Annotate/ground_truth.json")


def load_json_array(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Failed to read '{path}': {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"'{path}' is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})."
        ) from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Expected top-level JSON array in '{path}', got {type(data).__name__}."
        )

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Expected object at array index {index} in '{path}', "
                f"got {type(item).__name__}."
            )
        rows.append(item)

    return rows


def normalize_title(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def build_row_key(item: dict[str, Any]) -> str:
    title_key = normalize_title(item.get("title"))
    synopsis_key = normalize_text(item.get("synopsis"))
    return f"{title_key}\n{synopsis_key}"


def build_row_map(
    items: list[dict[str, Any]], path: Path
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    rows_by_key: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []

    for index, item in enumerate(items, start=1):
        title = item.get("title")
        title_key = normalize_title(title)
        if not title_key:
            raise ValueError(f"Missing/invalid title at row {index} in '{path}'.")
        row_key = build_row_key(item)
        if row_key in rows_by_key:
            raise ValueError(
                f"Duplicate title/synopsis pair for '{title}' found in '{path}' "
                f"at row {index}."
            )
        rows_by_key[row_key] = item
        ordered_keys.append(row_key)

    return rows_by_key, ordered_keys


def align_rows(
    initial_items: list[dict[str, Any]],
    second_items: list[dict[str, Any]],
    initial_path: Path,
    second_path: Path,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[str], list[str]]:
    initial_map, initial_order = build_row_map(initial_items, initial_path)
    second_map, second_order = build_row_map(second_items, second_path)

    missing_from_second = [title for title in initial_order if title not in second_map]
    missing_from_initial = [title for title in second_order if title not in initial_map]
    if missing_from_initial or missing_from_second:
        raise ValueError(
            "The files do not contain the same movie titles. "
            f"Missing from second: {len(missing_from_second)}. "
            f"Missing from initial: {len(missing_from_initial)}."
        )

    aligned = [(initial_map[row_key], second_map[row_key]) for row_key in initial_order]
    return aligned, initial_order, second_order


def compare_annotations(
    aligned_rows: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, str]]:
    disagreements: list[dict[str, str]] = []
    for initial_row, second_row in aligned_rows:
        initial_genre = str(initial_row.get("manual_genre", "")).strip()
        second_genre = str(second_row.get("manual_genre", "")).strip()
        if initial_genre == second_genre:
            continue

        disagreements.append(
            {
                "title": str(initial_row.get("title", "")),
                "initial_genre": initial_genre,
                "second_genre": second_genre,
            }
        )

    return disagreements


def format_counter(counter: Counter[str], header: str) -> str:
    lines = [header]
    if not counter:
        lines.append("  None")
        return "\n".join(lines)

    for genre, count in counter.most_common():
        lines.append(f"  {genre}: {count}")
    return "\n".join(lines)


def format_transition_counter(counter: Counter[tuple[str, str]]) -> str:
    lines = ["Disagreement transitions:"]
    if not counter:
        lines.append("  None")
        return "\n".join(lines)

    for (initial_genre, second_genre), count in counter.most_common():
        lines.append(f"  {initial_genre} -> {second_genre}: {count}")
    return "\n".join(lines)


def format_table(rows: list[dict[str, str]]) -> str:
    headers = ("Movie Title", "Initial Annotation", "Second Pass Annotation")
    title_width = max(len(headers[0]), *(len(row["title"]) for row in rows)) if rows else len(headers[0])
    initial_width = max(
        len(headers[1]), *(len(row["initial_genre"]) for row in rows)
    ) if rows else len(headers[1])
    second_width = max(
        len(headers[2]), *(len(row["second_genre"]) for row in rows)
    ) if rows else len(headers[2])

    divider = (
        f"+-{'-' * title_width}-+-{'-' * initial_width}-+-{'-' * second_width}-+"
    )
    header_line = (
        f"| {headers[0].ljust(title_width)} "
        f"| {headers[1].ljust(initial_width)} "
        f"| {headers[2].ljust(second_width)} |"
    )

    lines = [divider, header_line, divider]
    for row in rows:
        lines.append(
            f"| {row['title'].ljust(title_width)} "
            f"| {row['initial_genre'].ljust(initial_width)} "
            f"| {row['second_genre'].ljust(second_width)} |"
        )
    lines.append(divider)
    return "\n".join(lines)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ValueError(f"Failed to write '{path}': {exc}") from exc


def rows_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left == right


def validate_resume_output(
    existing_rows: list[dict[str, Any]],
    aligned_rows: list[tuple[dict[str, Any], dict[str, Any]]],
    output_path: Path,
) -> list[dict[str, Any]]:
    if len(existing_rows) > len(aligned_rows):
        raise ValueError(
            f"Resume file '{output_path}' has more rows than the inputs: "
            f"{len(existing_rows)} > {len(aligned_rows)}."
        )

    validated_prefix: list[dict[str, Any]] = []
    for index, existing_row in enumerate(existing_rows, start=1):
        initial_row, second_row = aligned_rows[index - 1]
        if rows_equal(initial_row, second_row):
            if not rows_equal(existing_row, initial_row):
                raise ValueError(
                    f"Resume mismatch at row {index} for title "
                    f"{initial_row.get('title')!r}: expected agreed row."
                )
        elif not (
            rows_equal(existing_row, initial_row) or rows_equal(existing_row, second_row)
        ):
            raise ValueError(
                f"Resume mismatch at row {index} for title "
                f"{initial_row.get('title')!r}: row does not match either source."
            )
        validated_prefix.append(existing_row)

    return validated_prefix


def prompt_for_choice(
    index: int,
    total: int,
    title: str,
    synopsis: str,
    initial_genre: str,
    second_genre: str,
) -> str:
    print()
    print(f"[{index}/{total}] {title}")
    print("Synopsis:")
    print(synopsis)
    print(f"1. Initial annotation: {initial_genre}")
    print(f"2. Second-pass annotation: {second_genre}")
    print("Choose [1/2], or q to save and quit.")

    while True:
        try:
            choice = input("> ").strip().lower()
        except EOFError:
            return "q"

        if choice in {"1", "2", "q"}:
            return choice
        print("Invalid choice. Enter 1, 2, or q.")


def run_report(args: argparse.Namespace) -> int:
    if args.max_rows is not None and args.max_rows < 0:
        print("Error: --max-rows must be >= 0.", file=sys.stderr)
        return 1

    try:
        initial_items = load_json_array(args.initial_json)
        second_items = load_json_array(args.second_json)
        aligned_rows, initial_order, second_order = align_rows(
            initial_items, second_items, args.initial_json, args.second_json
        )
        disagreements = compare_annotations(aligned_rows)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    initial_counts = Counter(row["initial_genre"] for row in disagreements)
    second_counts = Counter(row["second_genre"] for row in disagreements)
    transition_counts = Counter(
        (row["initial_genre"], row["second_genre"]) for row in disagreements
    )

    display_rows = disagreements
    if args.max_rows is not None:
        display_rows = disagreements[: args.max_rows]

    print(f"Initial file: {args.initial_json}")
    print(f"Second-pass file: {args.second_json}")
    print(f"Movies compared: {len(initial_order)}")
    print(f"Disagreements found: {len(disagreements)}")
    print()
    print(format_counter(initial_counts, "Disagreements by initial genre:"))
    print()
    print(format_counter(second_counts, "Disagreements by second-pass genre:"))
    print()
    print(format_transition_counter(transition_counts))
    print()
    if args.max_rows is not None and args.max_rows < len(disagreements):
        print(f"Showing first {len(display_rows)} disagreement row(s):")
    else:
        print("Disagreement table:")
    print(format_table(display_rows))
    return 0


def run_adjudicate(args: argparse.Namespace) -> int:
    if args.resume and args.restart:
        print("Error: use only one of --resume or --restart.", file=sys.stderr)
        return 1

    try:
        initial_items = load_json_array(args.initial_json)
        second_items = load_json_array(args.second_json)
        aligned_rows, _, _ = align_rows(
            initial_items, second_items, args.initial_json, args.second_json
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_path = args.output
    adjudicated_rows: list[dict[str, Any]] = []

    if output_path.exists():
        if args.restart:
            adjudicated_rows = []
        elif args.resume:
            try:
                existing_rows = load_json_array(output_path)
                adjudicated_rows = validate_resume_output(
                    existing_rows, aligned_rows, output_path
                )
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
        else:
            print(
                "Error: output file already exists. Use --resume to continue or "
                "--restart to overwrite it.",
                file=sys.stderr,
            )
            return 1

    total_rows = len(aligned_rows)
    start_index = len(adjudicated_rows)

    if start_index == total_rows:
        print(f"Adjudication already complete in: {output_path}")
        print(f"Rows: {total_rows}")
        return 0

    manual_total = sum(
        1 for initial_row, second_row in aligned_rows if initial_row != second_row
    )
    manual_done = sum(
        1
        for index, row in enumerate(adjudicated_rows)
        if not rows_equal(aligned_rows[index][0], aligned_rows[index][1])
    )

    print(f"Initial file: {args.initial_json}")
    print(f"Second-pass file: {args.second_json}")
    print(f"Output file: {output_path}")
    print(f"Rows total: {total_rows}")
    print(f"Rows already written: {start_index}")
    print(f"Disagreements requiring review: {manual_total}")

    try:
        for row_index in range(start_index, total_rows):
            initial_row, second_row = aligned_rows[row_index]

            if rows_equal(initial_row, second_row):
                adjudicated_rows.append(dict(initial_row))
                write_json(output_path, adjudicated_rows)
                continue

            title = str(initial_row.get("title", ""))
            synopsis = str(initial_row.get("synopsis", ""))
            initial_genre = str(initial_row.get("manual_genre", "")).strip()
            second_genre = str(second_row.get("manual_genre", "")).strip()
            choice = prompt_for_choice(
                index=row_index + 1,
                total=total_rows,
                title=title,
                synopsis=synopsis,
                initial_genre=initial_genre,
                second_genre=second_genre,
            )

            if choice == "q":
                write_json(output_path, adjudicated_rows)
                next_row = len(adjudicated_rows) + 1
                print()
                print(f"Progress saved to: {output_path}")
                print(
                    "Resume with: "
                    f"python3 disagreement.py adjudicate --resume -o {output_path!s}"
                )
                print(f"Next row: {next_row}")
                print(f"Manual disagreements reviewed: {manual_done} / {manual_total}")
                return 0

            if choice == "1":
                adjudicated_rows.append(dict(initial_row))
            else:
                adjudicated_rows.append(dict(second_row))
            manual_done += 1
            write_json(output_path, adjudicated_rows)
    except KeyboardInterrupt:
        try:
            write_json(output_path, adjudicated_rows)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print()
        print(f"Progress saved to: {output_path}")
        print(
            "Resume with: "
            f"python3 disagreement.py adjudicate --resume -o {output_path!s}"
        )
        return 130
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print()
    print("Adjudication complete.")
    print(f"Output written to: {output_path}")
    print(f"Manual disagreements reviewed: {manual_done} / {manual_total}")
    return 0


def build_report_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "initial_json",
        nargs="?",
        type=Path,
        default=DEFAULT_INITIAL,
        help=f"Initial annotation JSON path. Defaults to '{DEFAULT_INITIAL}'.",
    )
    parser.add_argument(
        "second_json",
        nargs="?",
        type=Path,
        default=DEFAULT_SECOND,
        help=f"Second-pass annotation JSON path. Defaults to '{DEFAULT_SECOND}'.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on displayed disagreement table rows.",
    )


def build_adjudicate_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "initial_json",
        nargs="?",
        type=Path,
        default=DEFAULT_INITIAL,
        help=f"Initial annotation JSON path. Defaults to '{DEFAULT_INITIAL}'.",
    )
    parser.add_argument(
        "second_json",
        nargs="?",
        type=Path,
        default=DEFAULT_SECOND,
        help=f"Second-pass annotation JSON path. Defaults to '{DEFAULT_SECOND}'.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output ground-truth JSON path. Defaults to '{DEFAULT_OUTPUT}'.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output file.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart adjudication and overwrite the output file.",
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in {"report", "adjudicate"}:
        root_parser = argparse.ArgumentParser(
            description="Compare annotation files or manually adjudicate disagreements."
        )
        subparsers = root_parser.add_subparsers(dest="command", required=True)

        report_parser = subparsers.add_parser(
            "report", help="Show disagreement counts and a disagreement table."
        )
        build_report_parser(report_parser)

        adjudicate_parser = subparsers.add_parser(
            "adjudicate",
            help="Interactively choose between initial and second-pass rows.",
        )
        build_adjudicate_parser(adjudicate_parser)

        args = root_parser.parse_args(argv)
        if args.command == "report":
            return run_report(args)
        return run_adjudicate(args)

    parser = argparse.ArgumentParser(
        description="Compare two annotation files and report genre disagreements."
    )
    build_report_parser(parser)
    args = parser.parse_args(argv)
    return run_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
