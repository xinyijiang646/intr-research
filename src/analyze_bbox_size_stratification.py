import csv
from collections import Counter
from pathlib import Path
from statistics import mean, median


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_PATH = (
    RESULTS_DIR
    / "formal_400_bbox_enrichment.csv"
)


def load_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def str_to_bool(value):
    return (
        value.strip().lower()
        == "true"
    )


def percentile(values, q):
    values = sorted(values)

    position = q * (len(values) - 1)

    lower_index = int(position)

    upper_index = min(
        lower_index + 1,
        len(values) - 1,
    )

    fraction = (
        position - lower_index
    )

    return (
        values[lower_index]
        + fraction
        * (
            values[upper_index]
            - values[lower_index]
        )
    )


def quartile_boundaries(rows):
    values = sorted(
        float(
            row["bbox_area_ratio"]
        )
        for row in rows
    )

    q1 = percentile(
        values,
        0.25,
    )

    q2 = percentile(
        values,
        0.50,
    )

    q3 = percentile(
        values,
        0.75,
    )

    return q1, q2, q3


def assign_quartile(
    area_ratio,
    q1,
    q2,
    q3,
):
    if area_ratio <= q1:
        return "Q1_smallest"

    if area_ratio <= q2:
        return "Q2"

    if area_ratio <= q3:
        return "Q3"

    return "Q4_largest"


def category_metrics(
    rows,
    category_key,
):
    counts = Counter(
        row[category_key]
        for row in rows
    )

    total = len(rows)

    prediction_stable = (
        counts["A"]
        + counts["B"]
    )

    attention_unstable = (
        counts["B"]
        + counts["D"]
    )

    if prediction_stable > 0:
        b_among_stable = (
            counts["B"]
            / prediction_stable
        )
    else:
        b_among_stable = 0.0

    return {
        "A": counts["A"],
        "B": counts["B"],
        "C": counts["C"],
        "D": counts["D"],
        "prediction_stable":
            prediction_stable / total,
        "attention_unstable":
            attention_unstable / total,
        "b_all":
            counts["B"] / total,
        "b_among_stable":
            b_among_stable,
    }


if __name__ == "__main__":

    rows = load_csv(
        INPUT_PATH
    )

    print(
        "=== BBox Size Stratification ==="
    )

    print(
        "Input rows:",
        len(rows),
    )

    assert len(rows) == 400

    # Quartile boundaries
    q1, q2, q3 = (
        quartile_boundaries(
            rows
        )
    )

    print()
    print(
        "=== BBox Area Quartile Boundaries ==="
    )

    print(
        f"Q1 boundary (25%): {q1:.4f}"
    )

    print(
        f"Q2 boundary (50%): {q2:.4f}"
    )

    print(
        f"Q3 boundary (75%): {q3:.4f}"
    )

    # Assign quartiles
    quartile_rows = {
        "Q1_smallest": [],
        "Q2": [],
        "Q3": [],
        "Q4_largest": [],
    }

    for row in rows:

        area_ratio = float(
            row[
                "bbox_area_ratio"
            ]
        )

        quartile = (
            assign_quartile(
                area_ratio,
                q1,
                q2,
                q3,
            )
        )

        quartile_rows[
            quartile
        ].append(
            row
        )

    print()
    print(
        "=== Quartile Size Check ==="
    )

    for quartile, group in (
        quartile_rows.items()
    ):
        print(
            quartile,
            len(group),
        )

    # Summary by quartile
    print()
    print(
        "=== Quartile Descriptive Summary ==="
    )

    for quartile, group in (
        quartile_rows.items()
    ):

        area_values = [
            float(
                row[
                    "bbox_area_ratio"
                ]
            )
            for row in group
        ]

        print()
        print(
            quartile
        )

        print(
            f"  n: {len(group)}"
        )

        print(
            f"  mean bbox area ratio: "
            f"{mean(area_values):.4f}"
        )

        print(
            f"  median bbox area ratio: "
            f"{median(area_values):.4f}"
        )

    # Blur stratification
    print()
    print(
        "=== Background Blur by BBox Size ==="
    )

    print(
        "Quartile       "
        "A    B    C    D    "
        "PredStable  AttnUnstable  "
        "B/all  B/stable"
    )

    for quartile, group in (
        quartile_rows.items()
    ):

        metrics = (
            category_metrics(
                group,
                "blur_category",
            )
        )

        print(
            f"{quartile:13s} "
            f"{metrics['A']:3d} "
            f"{metrics['B']:3d} "
            f"{metrics['C']:3d} "
            f"{metrics['D']:3d}   "
            f"{metrics['prediction_stable']:.1%}      "
            f"{metrics['attention_unstable']:.1%}       "
            f"{metrics['b_all']:.1%}   "
            f"{metrics['b_among_stable']:.1%}"
        )

    # Mask stratification
    print()
    print(
        "=== Background Mask by BBox Size ==="
    )

    print(
        "Quartile       "
        "A    B    C    D    "
        "PredStable  AttnUnstable  "
        "B/all  B/stable"
    )

    for quartile, group in (
        quartile_rows.items()
    ):

        metrics = (
            category_metrics(
                group,
                "mask_category",
            )
        )

        print(
            f"{quartile:13s} "
            f"{metrics['A']:3d} "
            f"{metrics['B']:3d} "
            f"{metrics['C']:3d} "
            f"{metrics['D']:3d}   "
            f"{metrics['prediction_stable']:.1%}      "
            f"{metrics['attention_unstable']:.1%}       "
            f"{metrics['b_all']:.1%}   "
            f"{metrics['b_among_stable']:.1%}"
        )

    # Persistent B rate
    print()
    print(
        "=== Persistent B by BBox Size ==="
    )

    print(
        "Quartile       "
        "PersistentB   Rate"
    )

    for quartile, group in (
        quartile_rows.items()
    ):

        persistent_b_count = sum(
            1
            for row in group
            if str_to_bool(
                row[
                    "persistent_B"
                ]
            )
        )

        rate = (
            persistent_b_count
            / len(group)
        )

        print(
            f"{quartile:13s} "
            f"{persistent_b_count:3d}/"
            f"{len(group):3d}      "
            f"{rate:.1%}"
        )

    # Persistent A rate
    print()
    print(
        "=== Persistent A by BBox Size ==="
    )

    print(
        "Quartile       "
        "PersistentA   Rate"
    )

    for quartile, group in (
        quartile_rows.items()
    ):

        persistent_a_count = sum(
            1
            for row in group
            if str_to_bool(
                row[
                    "persistent_A"
                ]
            )
        )

        rate = (
            persistent_a_count
            / len(group)
        )

        print(
            f"{quartile:13s} "
            f"{persistent_a_count:3d}/"
            f"{len(group):3d}      "
            f"{rate:.1%}"
        )

    # Extra continuous metrics by quartile
    print()
    print(
        "=== Attention Similarity by BBox Size ==="
    )

    for quartile, group in (
        quartile_rows.items()
    ):

        blur_cosine = [
            float(
                row[
                    "attention_cosine"
                ]
            )
            for row in []
        ]

    # We do not currently have blur/mask
    # cosine columns in the enrichment CSV.
    # Category-level stratification above
    # is therefore the primary analysis.

    print()
    print(
        "=== Analysis Complete ==="
    )