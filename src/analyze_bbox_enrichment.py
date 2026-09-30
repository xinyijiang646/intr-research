import csv
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_PATH = (
    RESULTS_DIR
    / "formal_400_bbox_attention.csv"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "formal_400_bbox_enrichment.csv"
)

BOOTSTRAP_SEED = 42
BOOTSTRAP_REPEATS = 10000


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

    position = (
        q * (len(values) - 1)
    )

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


def confidence_interval(
    values,
    alpha=0.05,
):
    return (
        percentile(
            values,
            alpha / 2,
        ),
        percentile(
            values,
            1 - alpha / 2,
        ),
    )


def summarize_group(
    name,
    rows,
    key,
):
    if not rows:
        print(
            f"{name}: no samples"
        )
        return

    values = [
        float(row[key])
        for row in rows
    ]

    print(
        f"{name:24s} | "
        f"n={len(values):3d} | "
        f"mean={mean(values):.4f} | "
        f"median={median(values):.4f}"
    )


def build_class_groups(rows):
    groups = defaultdict(list)

    for row in rows:
        class_id = int(
            row["class_id"]
        )

        groups[class_id].append(
            row
        )

    return groups


def mean_difference(
    group_a,
    group_b,
    key,
):
    """
    Returns:
        mean(group_a) - mean(group_b)
    """

    a_values = [
        float(row[key])
        for row in group_a
    ]

    b_values = [
        float(row[key])
        for row in group_b
    ]

    return (
        mean(a_values)
        - mean(b_values)
    )


def clustered_bootstrap_difference(
    rows,
    group_a_filter,
    group_b_filter,
    key,
    repeats=10000,
    seed=42,
):
    """
    Class-clustered bootstrap.

    Resample classes with replacement,
    preserving both formal samples within
    each selected class.
    """

    rng = random.Random(seed)

    class_groups = (
        build_class_groups(rows)
    )

    class_ids = sorted(
        class_groups.keys()
    )

    bootstrap_values = []

    for _ in range(repeats):

        sampled_rows = []

        sampled_classes = [
            rng.choice(class_ids)
            for _ in range(
                len(class_ids)
            )
        ]

        for class_id in sampled_classes:
            sampled_rows.extend(
                class_groups[class_id]
            )

        group_a = [
            row
            for row in sampled_rows
            if group_a_filter(row)
        ]

        group_b = [
            row
            for row in sampled_rows
            if group_b_filter(row)
        ]

        # Very unlikely to be empty here,
        # but keep the guard for robustness.
        if (
            not group_a
            or not group_b
        ):
            continue

        bootstrap_values.append(
            mean_difference(
                group_a,
                group_b,
                key,
            )
        )

    return bootstrap_values


if __name__ == "__main__":

    rows = load_csv(
        INPUT_PATH
    )

    print(
        "=== BBox Attention Enrichment Analysis ==="
    )

    print(
        "Input rows:",
        len(rows),
    )

    assert len(rows) == 400

    enriched_rows = []

    # Compute area-normalized enrichment
    for row in rows:

        bbox_attention_ratio = float(
            row[
                "bbox_attention_ratio"
            ]
        )

        bbox_feature_cells = int(
            row[
                "bbox_feature_cells"
            ]
        )

        total_feature_cells = int(
            row[
                "total_feature_cells"
            ]
        )

        assert total_feature_cells > 0

        bbox_area_ratio = (
            bbox_feature_cells
            / total_feature_cells
        )

        assert bbox_area_ratio > 0

        bbox_enrichment = (
            bbox_attention_ratio
            / bbox_area_ratio
        )

        enriched_row = dict(row)

        enriched_row[
            "bbox_area_ratio"
        ] = bbox_area_ratio

        enriched_row[
            "bbox_attention_enrichment"
        ] = bbox_enrichment

        enriched_rows.append(
            enriched_row
        )

    # Save
    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        fieldnames = list(
            enriched_rows[0].keys()
        )

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            enriched_rows
        )

    print()
    print(
        "Saved enrichment results to:"
    )

    print(
        OUTPUT_PATH
    )

    # Build groups
    blur_a = [
        row
        for row in enriched_rows
        if row[
            "blur_category"
        ] == "A"
    ]

    blur_b = [
        row
        for row in enriched_rows
        if row[
            "blur_category"
        ] == "B"
    ]

    mask_a = [
        row
        for row in enriched_rows
        if row[
            "mask_category"
        ] == "A"
    ]

    mask_b = [
        row
        for row in enriched_rows
        if row[
            "mask_category"
        ] == "B"
    ]

    persistent_a = [
        row
        for row in enriched_rows
        if str_to_bool(
            row[
                "persistent_A"
            ]
        )
    ]

    persistent_b = [
        row
        for row in enriched_rows
        if str_to_bool(
            row[
                "persistent_B"
            ]
        )
    ]

    # First inspect bbox area itself
    print()
    print(
        "=== BBox Area Ratio Summary ==="
    )

    summarize_group(
        "All formal images",
        enriched_rows,
        "bbox_area_ratio",
    )

    print()

    summarize_group(
        "Blur A",
        blur_a,
        "bbox_area_ratio",
    )

    summarize_group(
        "Blur B",
        blur_b,
        "bbox_area_ratio",
    )

    print()

    summarize_group(
        "Mask A",
        mask_a,
        "bbox_area_ratio",
    )

    summarize_group(
        "Mask B",
        mask_b,
        "bbox_area_ratio",
    )

    print()

    summarize_group(
        "Persistent A",
        persistent_a,
        "bbox_area_ratio",
    )

    summarize_group(
        "Persistent B",
        persistent_b,
        "bbox_area_ratio",
    )

    # Raw bbox attention ratio
    print()
    print(
        "=== Raw BBox Attention Ratio Summary ==="
    )

    summarize_group(
        "Persistent A",
        persistent_a,
        "bbox_attention_ratio",
    )

    summarize_group(
        "Persistent B",
        persistent_b,
        "bbox_attention_ratio",
    )

    # Area-normalized enrichment
    print()
    print(
        "=== BBox Attention Enrichment Summary ==="
    )

    summarize_group(
        "All formal images",
        enriched_rows,
        "bbox_attention_enrichment",
    )

    print()

    summarize_group(
        "Blur A",
        blur_a,
        "bbox_attention_enrichment",
    )

    summarize_group(
        "Blur B",
        blur_b,
        "bbox_attention_enrichment",
    )

    print()

    summarize_group(
        "Mask A",
        mask_a,
        "bbox_attention_enrichment",
    )

    summarize_group(
        "Mask B",
        mask_b,
        "bbox_attention_enrichment",
    )

    print()

    summarize_group(
        "Persistent A",
        persistent_a,
        "bbox_attention_enrichment",
    )

    summarize_group(
        "Persistent B",
        persistent_b,
        "bbox_attention_enrichment",
    )

    # Point differences
    print()
    print(
        "=== Mean Group Differences ==="
    )

    comparisons = [
        (
            "Blur A - Blur B",
            blur_a,
            blur_b,
        ),
        (
            "Mask A - Mask B",
            mask_a,
            mask_b,
        ),
        (
            "Persistent A - Persistent B",
            persistent_a,
            persistent_b,
        ),
    ]

    for name, group_a, group_b in comparisons:

        raw_diff = mean_difference(
            group_a,
            group_b,
            "bbox_attention_ratio",
        )

        area_diff = mean_difference(
            group_a,
            group_b,
            "bbox_area_ratio",
        )

        enrichment_diff = mean_difference(
            group_a,
            group_b,
            "bbox_attention_enrichment",
        )

        print()
        print(name)

        print(
            f"  Raw attention ratio diff: "
            f"{raw_diff:+.4f}"
        )

        print(
            f"  BBox area ratio diff: "
            f"{area_diff:+.4f}"
        )

        print(
            f"  Enrichment diff: "
            f"{enrichment_diff:+.4f}"
        )

    # Bootstrap CI for persistent A vs B
    print()
    print(
        "=== Persistent A vs B "
        "Class-Clustered Bootstrap ==="
    )

    persistent_a_filter = (
        lambda row:
            str_to_bool(
                row[
                    "persistent_A"
                ]
            )
    )

    persistent_b_filter = (
        lambda row:
            str_to_bool(
                row[
                    "persistent_B"
                ]
            )
    )

    for key in [
        "bbox_attention_ratio",
        "bbox_area_ratio",
        "bbox_attention_enrichment",
    ]:

        point_estimate = (
            mean_difference(
                persistent_a,
                persistent_b,
                key,
            )
        )

        bootstrap_values = (
            clustered_bootstrap_difference(
                rows=enriched_rows,
                group_a_filter=
                    persistent_a_filter,
                group_b_filter=
                    persistent_b_filter,
                key=key,
                repeats=
                    BOOTSTRAP_REPEATS,
                seed=
                    BOOTSTRAP_SEED,
            )
        )

        lower, upper = (
            confidence_interval(
                bootstrap_values
            )
        )

        print(
            f"{key:28s} | "
            f"diff "
            f"{point_estimate:+.4f} | "
            f"95% CI "
            f"[{lower:+.4f}, "
            f"{upper:+.4f}]"
        )

    print()
    print(
        "=== Analysis Complete ==="
    )
