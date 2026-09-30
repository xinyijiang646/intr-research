import csv
import random
from pathlib import Path
from statistics import mean, median

import numpy as np


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

BBOX_PATH = (
    RESULTS_DIR
    / "formal_400_bbox_enrichment.csv"
)

SEED = 42
N_BOOTSTRAP = 10000


def load_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def to_lookup(rows):
    return {
        int(row["image_id"]): row
        for row in rows
    }


def percentile(values, p):
    values = sorted(values)

    index = (
        (len(values) - 1)
        * p
    )

    lower = int(index)

    upper = min(
        lower + 1,
        len(values) - 1,
    )

    weight = (
        index - lower
    )

    return (
        values[lower]
        * (1.0 - weight)
        +
        values[upper]
        * weight
    )


def rankdata(values):
    """
    Average ranks for ties.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    ranks = np.empty(
        len(values),
        dtype=float,
    )

    i = 0

    while i < len(values):
        j = i

        while (
            j + 1 < len(values)
            and
            values[
                order[j + 1]
            ]
            ==
            values[
                order[i]
            ]
        ):
            j += 1

        average_rank = (
            i + j
        ) / 2.0 + 1.0

        for k in range(
            i,
            j + 1,
        ):
            ranks[
                order[k]
            ] = average_rank

        i = j + 1

    return ranks


def pearson(x, y):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    x_centered = (
        x - x.mean()
    )

    y_centered = (
        y - y.mean()
    )

    denominator = (
        np.sqrt(
            np.sum(
                x_centered ** 2
            )
        )
        *
        np.sqrt(
            np.sum(
                y_centered ** 2
            )
        )
    )

    if denominator == 0:
        return float("nan")

    return float(
        np.sum(
            x_centered
            * y_centered
        )
        /
        denominator
    )


def spearman(x, y):
    return pearson(
        rankdata(x),
        rankdata(y),
    )


def class_cluster_bootstrap_spearman(
    rows,
    x_key,
    y_key,
    n_bootstrap=10000,
    seed=42,
):
    rng = random.Random(seed)

    rows_by_class = {}

    for row in rows:
        rows_by_class.setdefault(
            row["class_id"],
            [],
        ).append(row)

    class_ids = sorted(
        rows_by_class
    )

    values = []

    for _ in range(
        n_bootstrap
    ):
        sampled_classes = [
            rng.choice(
                class_ids
            )
            for _ in class_ids
        ]

        sampled_rows = []

        for class_id in (
            sampled_classes
        ):
            sampled_rows.extend(
                rows_by_class[
                    class_id
                ]
            )

        x = [
            row[x_key]
            for row in sampled_rows
        ]

        y = [
            row[y_key]
            for row in sampled_rows
        ]

        rho = spearman(
            x,
            y,
        )

        if not np.isnan(rho):
            values.append(
                rho
            )

    return (
        percentile(
            values,
            0.025,
        ),
        percentile(
            values,
            0.975,
        ),
    )


def build_paired_rows(
    baseline_path,
    method_path,
    bbox_lookup,
):
    baseline = to_lookup(
        load_csv(
            baseline_path
        )
    )

    method = to_lookup(
        load_csv(
            method_path
        )
    )

    assert set(
        baseline
    ) == set(
        method
    )

    assert len(
        baseline
    ) == 400

    rows = []

    for image_id in sorted(
        baseline
    ):
        base = baseline[
            image_id
        ]

        meth = method[
            image_id
        ]

        bbox = bbox_lookup[
            image_id
        ]

        class_id = int(
            base["class_id"]
        )

        assert class_id == int(
            meth["class_id"]
        )

        assert class_id == int(
            bbox["class_id"]
        )

        rows.append(
            {
                "image_id":
                    image_id,

                "class_id":
                    class_id,

                "bbox_area_ratio":
                    float(
                        bbox[
                            "bbox_area_ratio"
                        ]
                    ),

                "delta_cosine":
                    (
                        float(
                            meth[
                                "attention_cosine"
                            ]
                        )
                        -
                        float(
                            base[
                                "attention_cosine"
                            ]
                        )
                    ),

                "delta_pearson":
                    (
                        float(
                            meth[
                                "attention_pearson"
                            ]
                        )
                        -
                        float(
                            base[
                                "attention_pearson"
                            ]
                        )
                    ),

                "delta_iou":
                    (
                        float(
                            meth[
                                "attention_top20_iou"
                            ]
                        )
                        -
                        float(
                            base[
                                "attention_top20_iou"
                            ]
                        )
                    ),

                "baseline_category":
                    base[
                        "category"
                    ],

                "method_category":
                    meth[
                        "category"
                    ],

                "baseline_prediction_stable":
                    (
                        base[
                            "prediction_unchanged"
                        ]
                        == "True"
                    ),

                "method_prediction_stable":
                    (
                        meth[
                            "prediction_unchanged"
                        ]
                        == "True"
                    ),
            }
        )

    return rows


def assign_quartiles(rows):
    areas = sorted(
        row["bbox_area_ratio"]
        for row in rows
    )

    q1 = percentile(
        areas,
        0.25,
    )

    q2 = percentile(
        areas,
        0.50,
    )

    q3 = percentile(
        areas,
        0.75,
    )

    print(
        "Quartile boundaries:",
        f"{q1:.6f},",
        f"{q2:.6f},",
        f"{q3:.6f}",
    )

    for row in rows:
        area = row[
            "bbox_area_ratio"
        ]

        if area <= q1:
            row["quartile"] = "Q1"
        elif area <= q2:
            row["quartile"] = "Q2"
        elif area <= q3:
            row["quartile"] = "Q3"
        else:
            row["quartile"] = "Q4"


def summarize_quartiles(rows):
    for quartile in [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    ]:
        group = [
            row
            for row in rows
            if row[
                "quartile"
            ] == quartile
        ]

        baseline_stable = [
            row
            for row in group
            if row[
                "baseline_prediction_stable"
            ]
        ]

        method_stable = [
            row
            for row in group
            if row[
                "method_prediction_stable"
            ]
        ]

        baseline_b = sum(
            row[
                "baseline_category"
            ] == "B"
            for row in group
        )

        method_b = sum(
            row[
                "method_category"
            ] == "B"
            for row in group
        )

        baseline_b_stable = (
            baseline_b
            /
            len(
                baseline_stable
            )
            if baseline_stable
            else float("nan")
        )

        method_b_stable = (
            method_b
            /
            len(
                method_stable
            )
            if method_stable
            else float("nan")
        )

        print()
        print(
            quartile,
            f"(n={len(group)})"
        )

        print(
            "  mean bbox area:",
            f"{mean(row['bbox_area_ratio'] for row in group):.6f}",
        )

        for key in [
            "delta_cosine",
            "delta_pearson",
            "delta_iou",
        ]:
            values = [
                row[key]
                for row in group
            ]

            print(
                f"  {key}:",
                f"mean {mean(values):+.6f},",
                f"median {median(values):+.6f}",
            )

        print(
            "  baseline B/stable:",
            f"{baseline_b}/"
            f"{len(baseline_stable)}",
            f"({baseline_b_stable:.1%})",
        )

        print(
            "  method B/stable:",
            f"{method_b}/"
            f"{len(method_stable)}",
            f"({method_b_stable:.1%})",
        )

        print(
            "  B/stable change:",
            f"{method_b_stable - baseline_b_stable:+.1%}",
        )


def summarize_continuous(rows):
    print()
    print(
        "--- Continuous scale association ---"
    )

    for key in [
        "delta_cosine",
        "delta_pearson",
        "delta_iou",
    ]:
        x = [
            row[
                "bbox_area_ratio"
            ]
            for row in rows
        ]

        y = [
            row[key]
            for row in rows
        ]

        rho = spearman(
            x,
            y,
        )

        low, high = (
            class_cluster_bootstrap_spearman(
                rows,
                "bbox_area_ratio",
                key,
                n_bootstrap=
                    N_BOOTSTRAP,
                seed=
                    SEED,
            )
        )

        print()
        print(
            key
        )

        print(
            "  Spearman rho:",
            f"{rho:+.6f}",
        )

        print(
            "  95% class-clustered "
            "bootstrap CI:",
            f"[{low:+.6f}, "
            f"{high:+.6f}]",
        )


def analyze(
    name,
    baseline_path,
    method_path,
    bbox_lookup,
):
    print()
    print(
        "=" * 70
    )

    print(name)

    print(
        "=" * 70
    )

    rows = build_paired_rows(
        baseline_path,
        method_path,
        bbox_lookup,
    )

    assign_quartiles(
        rows
    )

    summarize_quartiles(
        rows
    )

    summarize_continuous(
        rows
    )


if __name__ == "__main__":

    bbox_rows = load_csv(
        BBOX_PATH
    )

    bbox_lookup = to_lookup(
        bbox_rows
    )

    assert len(
        bbox_lookup
    ) == 400

    analyze(
        name="Background Blur",
        baseline_path=(
            RESULTS_DIR
            / "finetuned_lambda_0p0_background_blur.csv"
        ),
        method_path=(
            RESULTS_DIR
            / "finetuned_lambda_0p01_background_blur.csv"
        ),
        bbox_lookup=bbox_lookup,
    )

    analyze(
        name="Background Mask",
        baseline_path=(
            RESULTS_DIR
            / "finetuned_lambda_0p0_background_mask.csv"
        ),
        method_path=(
            RESULTS_DIR
            / "finetuned_lambda_0p01_background_mask.csv"
        ),
        bbox_lookup=bbox_lookup,
    )