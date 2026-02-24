import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np


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

UNKNOWN_LABEL = "Unknown"


# ----------------------------
# IO helpers
# ----------------------------
def load_json_array(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON array from a file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON array (list).")
    for i, item in enumerate(data[:5]):
        if not isinstance(item, dict):
            raise ValueError(f"{path} item {i} is not a JSON object (dict).")
    return data  # type: ignore[return-value]


# ----------------------------
# Normalization (titles + genres)
# ----------------------------
def normalize_title(raw: Any) -> str:
    """
    Normalize titles so minor formatting differences don't break alignment.
    - lowercases
    - trims leading/trailing spaces
    - collapses internal whitespace
    """
    if raw is None:
        return ""
    s = str(raw).strip().lower()
    # collapse repeated whitespace
    s = " ".join(s.split())
    return s


def normalize_genre(raw: Any) -> str:
    """Normalize/validate genre labels. Invalid/missing -> Unknown."""
    if raw is None:
        return UNKNOWN_LABEL
    if not isinstance(raw, str):
        return UNKNOWN_LABEL

    g = raw.strip()

    # Common variants you might see in manual annotations
    alias_map = {
        "Thriller/Crime": "Thriller / Crime",
        "Thriller /Crime": "Thriller / Crime",
        "Thriller/ Crime": "Thriller / Crime",
        "Sci-Fi": "Science Fiction",
        "Sci Fi": "Science Fiction",
        "Science-Fiction": "Science Fiction",
    }
    g = alias_map.get(g, g)

    if g in ALLOWED_GENRES:
        return g

    return UNKNOWN_LABEL


# ----------------------------
# Subject selection + rater maps
# ----------------------------
def first_n_titles(reference_items: List[Dict[str, Any]], n: int) -> List[str]:
    """
    Take the first n normalized titles from the reference file.
    These define the subject set and order.
    """
    titles: List[str] = []
    for obj in reference_items:
        t = normalize_title(obj.get("title"))
        if t:
            titles.append(t)
        if len(titles) >= n:
            break
    return titles


def build_title_to_genre_map(
    items: List[Dict[str, Any]],
    title_key: str = "title",
    genre_key: str = "manual_genre",
) -> Tuple[Dict[str, str], int, int]:
    """
    Build mapping: normalized_title -> normalized_genre.

    Returns:
      (map, missing_title_count, duplicate_title_count)
    """
    out: Dict[str, str] = {}
    missing_title = 0
    duplicates = 0
    seen = set()

    for obj in items:
        t_norm = normalize_title(obj.get(title_key))
        if not t_norm:
            missing_title += 1
            continue

        if t_norm in seen:
            duplicates += 1
        seen.add(t_norm)

        out[t_norm] = normalize_genre(obj.get(genre_key))

    return out, missing_title, duplicates


# ----------------------------
# Fleiss' Kappa computation
# ----------------------------
def build_contingency_matrix(
    subject_titles: List[str],
    rater_title_to_genre_maps: List[Dict[str, str]],
    categories: List[str],
) -> np.ndarray:
    """
    Rows = subjects, cols = categories, values = number of raters choosing each category.

    Missing title in a rater file -> Unknown, so each row sums to n_raters.
    """
    cat_index = {c: i for i, c in enumerate(categories)}
    n_subjects = len(subject_titles)
    n_categories = len(categories)
    n_raters = len(rater_title_to_genre_maps)

    M = np.zeros((n_subjects, n_categories), dtype=np.int64)

    for i, title in enumerate(subject_titles):
        for r_map in rater_title_to_genre_maps:
            g = r_map.get(title, UNKNOWN_LABEL)
            if g not in cat_index:
                g = UNKNOWN_LABEL
            M[i, cat_index[g]] += 1

    # Validate constant row sums (required by standard Fleiss)
    row_sums = M.sum(axis=1)
    if not np.all(row_sums == n_raters):
        bad = np.where(row_sums != n_raters)[0][:10]
        raise ValueError(
            f"Row sums are not constant = {n_raters}. "
            f"Example bad rows: {bad.tolist()} with sums {row_sums[bad].tolist()}."
        )

    return M


def calculate_fleiss_kappa(M: np.ndarray) -> float:
    """
    Standard Fleiss' Kappa for fixed number of raters per subject.
    M shape: (N subjects x K categories), each row sums to n (raters).
    """
    if M.ndim != 2:
        raise ValueError("Contingency matrix must be 2D.")

    N, K = M.shape
    if N == 0 or K == 0:
        raise ValueError("Contingency matrix must be non-empty.")

    n = int(M.sum(axis=1)[0])
    if n <= 1:
        raise ValueError("Fleiss' Kappa requires at least 2 raters per subject.")

    if not np.all(M.sum(axis=1) == n):
        raise ValueError("All subjects must have the same number of ratings (row sums constant).")

    # p_j = proportion of all assignments to category j
    p_j = M.sum(axis=0) / (N * n)

    # P_i = per-subject agreement
    P_i = (np.sum(M * (M - 1), axis=1)) / (n * (n - 1))

    P_bar = float(np.mean(P_i))
    P_e = float(np.sum(p_j**2))

    if np.isclose(1.0, P_e):
        return 0.0

    return float((P_bar - P_e) / (1.0 - P_e))


def interpret_kappa(kappa: float) -> str:
    if kappa < 0:
        return "Poor agreement (worse than chance)"
    if kappa < 0.2:
        return "Slight agreement"
    if kappa < 0.4:
        return "Fair agreement"
    if kappa < 0.6:
        return "Moderate agreement"
    if kappa < 0.8:
        return "Substantial agreement"
    return "Almost perfect agreement"


# ----------------------------
# Analysis runner
# ----------------------------
def analyze_agreement(
    name: str,
    comparison_files: List[str],
    reference_file: str,
    annotation_dir: Path,
    n_items: int = 75,
) -> None:
    print(f"\n{'='*70}")
    print(f"Analysis: {name}")
    print(f"{'='*70}")

    ref_path = annotation_dir / reference_file
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")

    ref_items = load_json_array(ref_path)
    subject_titles = first_n_titles(ref_items, n_items)
    if len(subject_titles) < n_items:
        print(f"⚠️  Warning: Reference file yielded only {len(subject_titles)} titles (requested {n_items}).")

    # Load rater maps
    rater_maps: List[Dict[str, str]] = []
    for fn in comparison_files:
        p = annotation_dir / fn
        if not p.exists():
            raise FileNotFoundError(f"Annotator file not found: {p}")
        items = load_json_array(p)
        tmap, missing_title, dup_titles = build_title_to_genre_map(items)
        rater_maps.append(tmap)

        msg = f"✓ Loaded {fn}: {len(items)} items, {len(tmap)} title-keys"
        if missing_title:
            msg += f", missing-title rows={missing_title}"
        if dup_titles:
            msg += f", duplicate titles={dup_titles} (kept last occurrence)"
        print(msg)

    categories = ALLOWED_GENRES

    # Filter subjects: only include those where all raters have a valid (non-Unknown) genre
    valid_subject_titles = []
    for title in subject_titles:
        all_valid = True
        for r_map in rater_maps:
            g = r_map.get(title, UNKNOWN_LABEL)
            if g == UNKNOWN_LABEL or g not in ALLOWED_GENRES:
                all_valid = False
                break
        if all_valid:
            valid_subject_titles.append(title)

    if len(valid_subject_titles) == 0:
        print("Error: No subjects with valid genres from all raters.")
        return

    M = build_contingency_matrix(valid_subject_titles, rater_maps, categories)
    kappa = calculate_fleiss_kappa(M)

    print(f"\nSubjects scored: {M.shape[0]} (filtered to exclude Unknown/invalid genres)")
    print(f"Subjects excluded: {len(subject_titles) - len(valid_subject_titles)}")
    print(f"Raters: {len(rater_maps)}")
    print(f"Categories: {M.shape[1]}")
    print(f"Fleiss' Kappa: {kappa:.4f}")
    print(f"Interpretation: {interpret_kappa(kappa)}")

    # Distribution
    counts = M.sum(axis=0).astype(int)
    print("\nGenre distribution across all ratings (subjects × raters):")
    for cat, c in zip(categories, counts):
        print(f"  {cat}: {int(c)}")


def main() -> None:
    annotation_dir = Path("data/annotation")

    print("\nFLEISS' KAPPA INTER-RATER AGREEMENT ANALYSIS")
    print("=" * 70)

    analyze_agreement(
        name="First 75 items (range 1-250) — aligned by title",
        comparison_files=[
            "first_75_1_250_annotated_T.json",
            "first_75_1_250_H.json",
            "1_250_O.json",
        ],
        reference_file="1_250_O.json",
        annotation_dir=annotation_dir,
        n_items=75,
    )

    analyze_agreement(
        name="First 75 items (range 251-500) — aligned by title",
        comparison_files=[
            "first_75_251_500_O.json",
            "alvin_annotated_first_75_251_500.json",
            "top_1000_movies_251_500_annotated_T.json",
        ],
        reference_file="top_1000_movies_251_500_annotated_T.json",
        annotation_dir=annotation_dir,
        n_items=75,
    )

    print(f"\n{'='*70}")
    print("Analysis Complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()