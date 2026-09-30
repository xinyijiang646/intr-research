import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

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


def to_float(row, key):
    return float(row[key])


def to_int(row, key):
    return int(row[key])


def percentile(values, q):
    """
    Simple percentile using linear interpolation.
    q should be between 0 and 1.
    """

    values = sorted(values)

    if not values:
        raise ValueError("Empty values")

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

    lower_value = values[
        lower_index
    ]

    upper_value = values[
        upper_index
    ]

    return (
        lower_value
        + fraction
        * (
            upper_value
            - lower_value
        )
    )


def confidence_interval(
    values,
    alpha=0.05,
):
    lower = percentile(
        values,
        alpha / 2,
    )

    upper = percentile(
        values,
        1 - alpha / 2,
    )

    return lower, upper


def category_summary(rows):
    counts = Counter(
        row["category"]
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
        "counts": counts,
        "total": total,
        "prediction_stable":
            prediction_stable,
        "attention_unstable":
            attention_unstable,
        "b_among_stable":
            b_among_stable,
    }


def metric_summary(rows, key):
    values = [
        to_float(row, key)
        for row in rows
    ]

    return {
        "mean": mean(values),
        "median": median(values),
    }


def build_by_id(rows):
    return {
        to_int(row, "image_id"): row
        for row in rows
    }


def build_class_groups(rows):
    """
    class_id -> list of rows
    """

    groups = defaultdict(list)

    for row in rows:
        class_id = to_int(
            row,
            "class_id",
        )

        groups[class_id].append(
            row
        )

    return groups


def bootstrap_clustered_single(
    rows,
    statistic_fn,
    repeats=10000,
    seed=42,
):
    """
    Class-clustered bootstrap.

    Resample classes with replacement.
    When a class is selected, include all
    rows belonging to that class.
    """

    rng = random.Random(seed)

    groups = build_class_groups(
        rows
    )

    class_ids = sorted(
        groups.keys()
    )

    n_classes = len(
        class_ids
    )

    bootstrap_values = []

    for _ in range(repeats):

        sampled_rows = []

        sampled_classes = [
            rng.choice(class_ids)
            for _ in range(
                n_classes
            )
        ]

        for class_id in sampled_classes:
            sampled_rows.extend(
                groups[class_id]
            )

        value = statistic_fn(
            sampled_rows
        )

        bootstrap_values.append(
            value
        )

    return bootstrap_values


def bootstrap_clustered_paired(
    blur_rows,
    mask_rows,
    statistic_fn,
    repeats=10000,
    seed=42,
):
    """
    Paired class-clustered bootstrap.

    Same sampled classes are used for both
    perturbations in each bootstrap replicate.
    """

    rng = random.Random(seed)

    blur_groups = build_class_groups(
        blur_rows
    )

    mask_groups = build_class_groups(
        mask_rows
    )

    class_ids = sorted(
        blur_groups.keys()
    )

    assert set(class_ids) == set(
        mask_groups.keys()
    )

    bootstrap_values = []

    for _ in range(repeats):

        sampled_blur = []
        sampled_mask = []

        sampled_classes = [
            rng.choice(class_ids)
            for _ in range(
                len(class_ids)
            )
        ]

        for class_id in sampled_classes:

            sampled_blur.extend(
                blur_groups[
                    class_id
                ]
            )

            sampled_mask.extend(
                mask_groups[
                    class_id
                ]
            )

        value = statistic_fn(
            sampled_blur,
            sampled_mask,
        )

        bootstrap_values.append(
            value
        )

    return bootstrap_values


def prediction_stable_rate(rows):
    summary = category_summary(
        rows
    )

    return (
        summary[
            "prediction_stable"
        ]
        / summary["total"]
    )


def attention_unstable_rate(rows):
    summary = category_summary(
        rows
    )

    return (
        summary[
            "attention_unstable"
        ]
        / summary["total"]
    )


def b_over_all_rate(rows):
    counts = Counter(
        row["category"]
        for row in rows
    )

    return (
        counts["B"]
        / len(rows)
    )


def b_among_prediction_stable_rate(
    rows
):
    summary = category_summary(
        rows
    )

    return summary[
        "b_among_stable"
    ]


def mean_metric(rows, key):
    return mean(
        to_float(row, key)
        for row in rows
    )


def paired_metric_difference(
    blur_rows,
    mask_rows,
    key,
):
    blur_by_id = build_by_id(
        blur_rows
    )

    mask_by_id = build_by_id(
        mask_rows
    )

    common_ids = sorted(
        set(blur_by_id)
        & set(mask_by_id)
    )

    differences = [
        (
            to_float(
                mask_by_id[image_id],
                key,
            )
            -
            to_float(
                blur_by_id[image_id],
                key,
            )
        )
        for image_id
        in common_ids
    ]

    return mean(
        differences
    )


def print_summary(
    name,
    rows,
):
    summary = category_summary(
        rows
    )

    counts = summary[
        "counts"
    ]

    total = summary[
        "total"
    ]

    print()
    print(
        f"=== {name} Category Summary ==="
    )

    for category in [
        "A",
        "B",
        "C",
        "D",
    ]:
        count = counts[
            category
        ]

        print(
            f"{category}: "
            f"{count}/{total} "
            f"({count / total:.1%})"
        )

    print(
        "Prediction stable:",
        f"{summary['prediction_stable']}"
        f"/{total}",
        f"({summary['prediction_stable'] / total:.1%})",
    )

    print(
        "Attention unstable:",
        f"{summary['attention_unstable']}"
        f"/{total}",
        f"({summary['attention_unstable'] / total:.1%})",
    )

    print(
        "B among prediction-stable:",
        f"{counts['B']}/"
        f"{summary['prediction_stable']}",
        f"({summary['b_among_stable']:.1%})",
    )


if __name__ == "__main__":

    blur_rows = load_csv(
        BLUR_PATH
    )

    mask_rows = load_csv(
        MASK_PATH
    )

    blur_by_id = build_by_id(
        blur_rows
    )

    mask_by_id = build_by_id(
        mask_rows
    )

    blur_ids = set(
        blur_by_id.keys()
    )

    mask_ids = set(
        mask_by_id.keys()
    )

    # Paired dataset integrity
    print(
        "=== Formal Paired Dataset Check ==="
    )

    print(
        "Blur rows:",
        len(blur_rows),
    )

    print(
        "Mask rows:",
        len(mask_rows),
    )

    print(
        "Blur unique images:",
        len(blur_ids),
    )

    print(
        "Mask unique images:",
        len(mask_ids),
    )

    print(
        "Same image set:",
        blur_ids == mask_ids,
    )

    assert len(blur_rows) == 400
    assert len(mask_rows) == 400

    assert len(blur_ids) == 400
    assert len(mask_ids) == 400

    assert blur_ids == mask_ids

    # Check classes also match
    for image_id in blur_ids:

        assert (
            to_int(
                blur_by_id[image_id],
                "class_id",
            )
            ==
            to_int(
                mask_by_id[image_id],
                "class_id",
            )
        )

    # Category summaries
    print_summary(
        "Background Blur",
        blur_rows,
    )

    print_summary(
        "Background Mask",
        mask_rows,
    )

    # Continuous metric summaries
    metrics = [
        "attention_cosine",
        "attention_pearson",
        "attention_top20_iou",
        "margin_change",
    ]

    print()
    print(
        "=== Continuous Metric Summary ==="
    )

    for key in metrics:

        blur_summary = (
            metric_summary(
                blur_rows,
                key,
            )
        )

        mask_summary = (
            metric_summary(
                mask_rows,
                key,
            )
        )

        print()

        print(key)

        print(
            "  Blur | "
            f"mean "
            f"{blur_summary['mean']:.4f} | "
            f"median "
            f"{blur_summary['median']:.4f}"
        )

        print(
            "  Mask | "
            f"mean "
            f"{mask_summary['mean']:.4f} | "
            f"median "
            f"{mask_summary['median']:.4f}"
        )

    # Category transitions
    transitions = Counter()

    for image_id in sorted(
        blur_ids
    ):

        blur_category = (
            blur_by_id[
                image_id
            ]["category"]
        )

        mask_category = (
            mask_by_id[
                image_id
            ]["category"]
        )

        transitions[
            (
                blur_category,
                mask_category,
            )
        ] += 1

    print()
    print(
        "=== Category Transitions: "
        "Blur -> Mask ==="
    )

    for source in [
        "A",
        "B",
        "C",
        "D",
    ]:
        for target in [
            "A",
            "B",
            "C",
            "D",
        ]:
            count = transitions[
                (
                    source,
                    target,
                )
            ]

            if count > 0:
                print(
                    f"{source} -> "
                    f"{target}: "
                    f"{count}"
                )

    # B-case overlap
    blur_b_ids = {
        image_id
        for image_id, row
        in blur_by_id.items()
        if row["category"] == "B"
    }

    mask_b_ids = {
        image_id
        for image_id, row
        in mask_by_id.items()
        if row["category"] == "B"
    }

    b_overlap = (
        blur_b_ids
        & mask_b_ids
    )

    b_union = (
        blur_b_ids
        | mask_b_ids
    )

    print()
    print(
        "=== Formal B-Case Overlap ==="
    )

    print(
        "Blur B cases:",
        len(blur_b_ids),
    )

    print(
        "Mask B cases:",
        len(mask_b_ids),
    )

    print(
        "B under both:",
        len(b_overlap),
    )

    print(
        "Fraction of blur-B "
        "also mask-B:",
        f"{len(b_overlap) / len(blur_b_ids):.1%}",
    )

    print(
        "Fraction of mask-B "
        "also blur-B:",
        f"{len(b_overlap) / len(mask_b_ids):.1%}",
    )

    print(
        "B-set Jaccard:",
        f"{len(b_overlap) / len(b_union):.3f}",
    )

    # Prediction stable under both
    blur_stable_ids = {
        image_id
        for image_id, row
        in blur_by_id.items()
        if row["category"]
        in {"A", "B"}
    }

    mask_stable_ids = {
        image_id
        for image_id, row
        in mask_by_id.items()
        if row["category"]
        in {"A", "B"}
    }

    stable_both = (
        blur_stable_ids
        & mask_stable_ids
    )

    unstable_both = {
        image_id
        for image_id
        in stable_both
        if (
            blur_by_id[
                image_id
            ]["category"] == "B"
            and
            mask_by_id[
                image_id
            ]["category"] == "B"
        )
    }

    print()
    print(
        "=== Prediction-Stable "
        "Under Both Perturbations ==="
    )

    print(
        "Stable under blur:",
        len(blur_stable_ids),
    )

    print(
        "Stable under mask:",
        len(mask_stable_ids),
    )

    print(
        "Stable under both:",
        len(stable_both),
    )

    print(
        "Attention unstable "
        "under both:",
        len(unstable_both),
    )

    if stable_both:
        print(
            "Persistent B among "
            "stable-under-both:",
            f"{len(unstable_both)}"
            f"/{len(stable_both)}",
            f"({len(unstable_both) / len(stable_both):.1%})",
        )

    # Paired metric differences
    print()
    print(
        "=== Paired Metric Differences: "
        "Mask - Blur ==="
    )

    for key in metrics:

        differences = []

        for image_id in sorted(
            blur_ids
        ):

            blur_value = to_float(
                blur_by_id[
                    image_id
                ],
                key,
            )

            mask_value = to_float(
                mask_by_id[
                    image_id
                ],
                key,
            )

            differences.append(
                mask_value
                - blur_value
            )

        print(
            f"{key:24s} | "
            f"mean delta "
            f"{mean(differences):+.4f} | "
            f"median delta "
            f"{median(differences):+.4f}"
        )

    # Single-perturbation clustered bootstrap CIs
    print()
    print(
        "=== Class-Clustered "
        "Bootstrap 95% CIs ==="
    )

    statistics = [
        (
            "Prediction stable",
            prediction_stable_rate,
        ),
        (
            "Attention unstable",
            attention_unstable_rate,
        ),
        (
            "B / all",
            b_over_all_rate,
        ),
        (
            "B among prediction-stable",
            b_among_prediction_stable_rate,
        ),
    ]

    for name, statistic_fn in statistics:

        print()
        print(name)

        for perturbation_name, rows in [
            (
                "Blur",
                blur_rows,
            ),
            (
                "Mask",
                mask_rows,
            ),
        ]:

            point_estimate = (
                statistic_fn(
                    rows
                )
            )

            bootstrap_values = (
                bootstrap_clustered_single(
                    rows=rows,
                    statistic_fn=statistic_fn,
                    repeats=BOOTSTRAP_REPEATS,
                    seed=BOOTSTRAP_SEED,
                )
            )

            lower, upper = (
                confidence_interval(
                    bootstrap_values
                )
            )

            print(
                f"  "
                f"{perturbation_name}: "
                f"{point_estimate:.1%} "
                f"[{lower:.1%}, "
                f"{upper:.1%}]"
            )

    # Paired clustered bootstrap:
    # Mask - Blur differences
    print()
    print(
        "=== Paired Class-Clustered "
        "Bootstrap: Mask - Blur ==="
    )

    paired_statistics = [
        (
            "Prediction stable rate",
            lambda blur, mask:
                prediction_stable_rate(mask)
                -
                prediction_stable_rate(blur),
        ),
        (
            "Attention unstable rate",
            lambda blur, mask:
                attention_unstable_rate(mask)
                -
                attention_unstable_rate(blur),
        ),
        (
            "B / all rate",
            lambda blur, mask:
                b_over_all_rate(mask)
                -
                b_over_all_rate(blur),
        ),
        (
            "B among stable rate",
            lambda blur, mask:
                b_among_prediction_stable_rate(mask)
                -
                b_among_prediction_stable_rate(blur),
        ),
    ]

    for name, statistic_fn in paired_statistics:

        point_estimate = (
            statistic_fn(
                blur_rows,
                mask_rows,
            )
        )

        bootstrap_values = (
            bootstrap_clustered_paired(
                blur_rows=blur_rows,
                mask_rows=mask_rows,
                statistic_fn=statistic_fn,
                repeats=BOOTSTRAP_REPEATS,
                seed=BOOTSTRAP_SEED,
            )
        )

        lower, upper = (
            confidence_interval(
                bootstrap_values
            )
        )

        print(
            f"{name:26s} | "
            f"delta "
            f"{point_estimate:+.1%} | "
            f"95% CI "
            f"[{lower:+.1%}, "
            f"{upper:+.1%}]"
        )

    # Paired clustered bootstrap for
    # continuous metric differences
    print()
    print(
        "=== Paired Metric Delta "
        "Bootstrap CIs ==="
    )

    for key in metrics:

        point_estimate = (
            paired_metric_difference(
                blur_rows,
                mask_rows,
                key,
            )
        )

        statistic_fn = (
            lambda blur, mask, metric=key:
                paired_metric_difference(
                    blur,
                    mask,
                    metric,
                )
        )

        bootstrap_values = (
            bootstrap_clustered_paired(
                blur_rows=blur_rows,
                mask_rows=mask_rows,
                statistic_fn=statistic_fn,
                repeats=BOOTSTRAP_REPEATS,
                seed=BOOTSTRAP_SEED,
            )
        )

        lower, upper = (
            confidence_interval(
                bootstrap_values
            )
        )

        print(
            f"{key:24s} | "
            f"delta "
            f"{point_estimate:+.4f} | "
            f"95% CI "
            f"[{lower:+.4f}, "
            f"{upper:+.4f}]"
        )

    print()
    print(
        "=== Analysis Complete ==="
    )
