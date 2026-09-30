import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

ENRICHMENT_PATH = (
    RESULTS_DIR
    / "formal_400_bbox_enrichment.csv"
)

BLUR_PATH = (
    RESULTS_DIR
    / "formal_400_background_blur.csv"
)

MASK_PATH = (
    RESULTS_DIR
    / "formal_400_background_mask.csv"
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


def build_by_id(rows):
    return {
        int(row["image_id"]): row
        for row in rows
    }


def str_to_bool(value):
    return (
        value.strip().lower()
        == "true"
    )


def rankdata(values):
    """
    Average ranks for tied values.
    Ranks start at 1.
    """

    indexed = sorted(
        enumerate(values),
        key=lambda item: item[1],
    )

    ranks = [0.0] * len(values)

    i = 0

    while i < len(indexed):

        j = i

        while (
            j + 1 < len(indexed)
            and
            indexed[j + 1][1]
            == indexed[i][1]
        ):
            j += 1

        average_rank = (
            (i + 1)
            + (j + 1)
        ) / 2.0

        for k in range(i, j + 1):
            original_index = (
                indexed[k][0]
            )

            ranks[
                original_index
            ] = average_rank

        i = j + 1

    return ranks


def pearson_correlation(
    x,
    y,
):
    assert len(x) == len(y)
    assert len(x) > 1

    x_mean = mean(x)
    y_mean = mean(y)

    numerator = sum(
        (a - x_mean)
        * (b - y_mean)
        for a, b
        in zip(x, y)
    )

    x_ss = sum(
        (a - x_mean) ** 2
        for a in x
    )

    y_ss = sum(
        (b - y_mean) ** 2
        for b in y
    )

    denominator = math.sqrt(
        x_ss * y_ss
    )

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


def spearman_correlation(
    x,
    y,
):
    x_ranks = rankdata(
        x
    )

    y_ranks = rankdata(
        y
    )

    return pearson_correlation(
        x_ranks,
        y_ranks,
    )


def percentile(values, q):
    values = sorted(values)

    position = (
        q
        * (len(values) - 1)
    )

    lower_index = int(
        position
    )

    upper_index = min(
        lower_index + 1,
        len(values) - 1,
    )

    fraction = (
        position
        - lower_index
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


def clustered_bootstrap_statistic(
    rows,
    statistic_fn,
    repeats=10000,
    seed=42,
):
    rng = random.Random(
        seed
    )

    groups = build_class_groups(
        rows
    )

    class_ids = sorted(
        groups.keys()
    )

    bootstrap_values = []

    for _ in range(repeats):

        sampled_rows = []

        sampled_classes = [
            rng.choice(
                class_ids
            )
            for _ in range(
                len(class_ids)
            )
        ]

        for class_id in sampled_classes:
            sampled_rows.extend(
                groups[
                    class_id
                ]
            )

        bootstrap_values.append(
            statistic_fn(
                sampled_rows
            )
        )

    return bootstrap_values


def spearman_from_rows(
    rows,
    x_key,
    y_key,
):
    x = [
        float(row[x_key])
        for row in rows
    ]

    y = [
        float(row[y_key])
        for row in rows
    ]

    return spearman_correlation(
        x,
        y,
    )


def persistent_b_binary_correlation(
    rows,
):
    x = [
        float(
            row[
                "bbox_area_ratio"
            ]
        )
        for row in rows
    ]

    y = [
        1.0
        if str_to_bool(
            row[
                "persistent_B"
            ]
        )
        else 0.0
        for row in rows
    ]

    return pearson_correlation(
        x,
        y,
    )


def persistent_b_area_difference(
    rows,
):
    persistent_b = [
        float(
            row[
                "bbox_area_ratio"
            ]
        )
        for row in rows
        if str_to_bool(
            row[
                "persistent_B"
            ]
        )
    ]

    non_persistent_b = [
        float(
            row[
                "bbox_area_ratio"
            ]
        )
        for row in rows
        if not str_to_bool(
            row[
                "persistent_B"
            ]
        )
    ]

    return (
        mean(
            persistent_b
        )
        -
        mean(
            non_persistent_b
        )
    )


if __name__ == "__main__":

    enrichment_rows = load_csv(
        ENRICHMENT_PATH
    )

    blur_rows = load_csv(
        BLUR_PATH
    )

    mask_rows = load_csv(
        MASK_PATH
    )

    enrichment_by_id = (
        build_by_id(
            enrichment_rows
        )
    )

    blur_by_id = build_by_id(
        blur_rows
    )

    mask_by_id = build_by_id(
        mask_rows
    )

    image_ids = set(
        enrichment_by_id
    )

    assert image_ids == set(
        blur_by_id
    )

    assert image_ids == set(
        mask_by_id
    )

    print(
        "=== Continuous Object-Scale Analysis ==="
    )

    print(
        "Images:",
        len(image_ids),
    )

    assert len(
        image_ids
    ) == 400

    # Merge into one analysis table
    merged_rows = []

    for image_id in sorted(
        image_ids
    ):

        enrichment_row = (
            enrichment_by_id[
                image_id
            ]
        )

        blur_row = blur_by_id[
            image_id
        ]

        mask_row = mask_by_id[
            image_id
        ]

        assert (
            int(
                enrichment_row[
                    "class_id"
                ]
            )
            ==
            int(
                blur_row[
                    "class_id"
                ]
            )
            ==
            int(
                mask_row[
                    "class_id"
                ]
            )
        )

        merged_rows.append(
            {
                "image_id":
                    image_id,

                "class_id":
                    int(
                        enrichment_row[
                            "class_id"
                        ]
                    ),

                "bbox_area_ratio":
                    float(
                        enrichment_row[
                            "bbox_area_ratio"
                        ]
                    ),

                "persistent_B":
                    enrichment_row[
                        "persistent_B"
                    ],

                "blur_cosine":
                    float(
                        blur_row[
                            "attention_cosine"
                        ]
                    ),

                "blur_iou":
                    float(
                        blur_row[
                            "attention_top20_iou"
                        ]
                    ),

                "mask_cosine":
                    float(
                        mask_row[
                            "attention_cosine"
                        ]
                    ),

                "mask_iou":
                    float(
                        mask_row[
                            "attention_top20_iou"
                        ]
                    ),
            }
        )

    # Continuous Spearman correlations
    analyses = [
        (
            "BBox area vs Blur cosine",
            "bbox_area_ratio",
            "blur_cosine",
        ),
        (
            "BBox area vs Blur IoU",
            "bbox_area_ratio",
            "blur_iou",
        ),
        (
            "BBox area vs Mask cosine",
            "bbox_area_ratio",
            "mask_cosine",
        ),
        (
            "BBox area vs Mask IoU",
            "bbox_area_ratio",
            "mask_iou",
        ),
    ]

    print()
    print(
        "=== Spearman Correlations ==="
    )

    for (
        name,
        x_key,
        y_key,
    ) in analyses:

        point_estimate = (
            spearman_from_rows(
                merged_rows,
                x_key,
                y_key,
            )
        )

        statistic_fn = (
            lambda rows,
            x=x_key,
            y=y_key:
                spearman_from_rows(
                    rows,
                    x,
                    y,
                )
        )

        bootstrap_values = (
            clustered_bootstrap_statistic(
                rows=merged_rows,
                statistic_fn=
                    statistic_fn,
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
            f"{name:28s} | "
            f"rho "
            f"{point_estimate:+.4f} | "
            f"95% CI "
            f"[{lower:+.4f}, "
            f"{upper:+.4f}]"
        )

    # Persistent B binary association
    print()
    print(
        "=== Persistent-B Association ==="
    )

    persistent_b_rows = [
        row
        for row in merged_rows
        if str_to_bool(
            row[
                "persistent_B"
            ]
        )
    ]

    non_persistent_b_rows = [
        row
        for row in merged_rows
        if not str_to_bool(
            row[
                "persistent_B"
            ]
        )
    ]

    persistent_b_areas = [
        float(
            row[
                "bbox_area_ratio"
            ]
        )
        for row in persistent_b_rows
    ]

    non_persistent_b_areas = [
        float(
            row[
                "bbox_area_ratio"
            ]
        )
        for row in non_persistent_b_rows
    ]

    print(
        "Persistent B count:",
        len(
            persistent_b_rows
        ),
    )

    print(
        "Non-persistent-B count:",
        len(
            non_persistent_b_rows
        ),
    )

    print()

    print(
        "Persistent B bbox area | "
        f"mean "
        f"{mean(persistent_b_areas):.4f} | "
        f"median "
        f"{median(persistent_b_areas):.4f}"
    )

    print(
        "Non-persistent-B bbox area | "
        f"mean "
        f"{mean(non_persistent_b_areas):.4f} | "
        f"median "
        f"{median(non_persistent_b_areas):.4f}"
    )

    # Point-biserial correlation is
    # Pearson correlation with a binary variable.
    binary_corr = (
        persistent_b_binary_correlation(
            merged_rows
        )
    )

    binary_bootstrap = (
        clustered_bootstrap_statistic(
            rows=merged_rows,
            statistic_fn=
                persistent_b_binary_correlation,
            repeats=
                BOOTSTRAP_REPEATS,
            seed=
                BOOTSTRAP_SEED,
        )
    )

    binary_lower, binary_upper = (
        confidence_interval(
            binary_bootstrap
        )
    )

    print()

    print(
        "BBox area vs Persistent-B "
        "indicator | "
        f"r "
        f"{binary_corr:+.4f} | "
        f"95% CI "
        f"[{binary_lower:+.4f}, "
        f"{binary_upper:+.4f}]"
    )

    # Mean area difference
    # Persistent B - Non-persistent B
    area_difference = (
        persistent_b_area_difference(
            merged_rows
        )
    )

    area_difference_bootstrap = (
        clustered_bootstrap_statistic(
            rows=merged_rows,
            statistic_fn=
                persistent_b_area_difference,
            repeats=
                BOOTSTRAP_REPEATS,
            seed=
                BOOTSTRAP_SEED,
        )
    )

    diff_lower, diff_upper = (
        confidence_interval(
            area_difference_bootstrap
        )
    )

    print(
        "Persistent-B minus "
        "non-persistent-B "
        "bbox area | "
        f"diff "
        f"{area_difference:+.4f} | "
        f"95% CI "
        f"[{diff_lower:+.4f}, "
        f"{diff_upper:+.4f}]"
    )

    print()
    print(
        "=== Analysis Complete ==="
    )
