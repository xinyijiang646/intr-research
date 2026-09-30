import csv
import random
from pathlib import Path
from statistics import mean, median


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


def percentile(
    values,
    p,
):
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


def class_cluster_bootstrap_ci(
    paired_rows,
    delta_key,
    n_bootstrap=10000,
    seed=42,
):
    rng = random.Random(seed)

    rows_by_class = {}

    for row in paired_rows:
        class_id = row["class_id"]

        rows_by_class.setdefault(
            class_id,
            [],
        ).append(
            row
        )

    class_ids = sorted(
        rows_by_class
    )

    bootstrap_means = []

    for _ in range(
        n_bootstrap
    ):
        sampled_classes = [
            rng.choice(
                class_ids
            )
            for _ in class_ids
        ]

        sampled_values = []

        for class_id in (
            sampled_classes
        ):
            sampled_values.extend(
                row[delta_key]
                for row in
                rows_by_class[
                    class_id
                ]
            )

        bootstrap_means.append(
            mean(
                sampled_values
            )
        )

    return (
        percentile(
            bootstrap_means,
            0.025,
        ),
        percentile(
            bootstrap_means,
            0.975,
        ),
    )


def analyze_pair(
    name,
    baseline_path,
    method_path,
):
    baseline_rows = (
        load_csv(
            baseline_path
        )
    )

    method_rows = (
        load_csv(
            method_path
        )
    )

    baseline_lookup = (
        to_lookup(
            baseline_rows
        )
    )

    method_lookup = (
        to_lookup(
            method_rows
        )
    )

    baseline_ids = set(
        baseline_lookup
    )

    method_ids = set(
        method_lookup
    )

    assert (
        baseline_ids
        ==
        method_ids
    )

    assert len(
        baseline_ids
    ) == 400

    paired_rows = []

    for image_id in sorted(
        baseline_ids
    ):
        baseline = (
            baseline_lookup[
                image_id
            ]
        )

        method = (
            method_lookup[
                image_id
            ]
        )

        assert (
            int(
                baseline[
                    "class_id"
                ]
            )
            ==
            int(
                method[
                    "class_id"
                ]
            )
        )

        paired_rows.append(
            {
                "image_id":
                    image_id,

                "class_id":
                    int(
                        baseline[
                            "class_id"
                        ]
                    ),

                "delta_cosine":
                    (
                        float(
                            method[
                                "attention_cosine"
                            ]
                        )
                        -
                        float(
                            baseline[
                                "attention_cosine"
                            ]
                        )
                    ),

                "delta_pearson":
                    (
                        float(
                            method[
                                "attention_pearson"
                            ]
                        )
                        -
                        float(
                            baseline[
                                "attention_pearson"
                            ]
                        )
                    ),

                "delta_iou":
                    (
                        float(
                            method[
                                "attention_top20_iou"
                            ]
                        )
                        -
                        float(
                            baseline[
                                "attention_top20_iou"
                            ]
                        )
                    ),
            }
        )

    print()
    print(
        "=" * 70
    )

    print(
        name
    )

    print(
        "=" * 70
    )

    for key in [
        "delta_cosine",
        "delta_pearson",
        "delta_iou",
    ]:
        values = [
            row[key]
            for row in paired_rows
        ]

        ci_low, ci_high = (
            class_cluster_bootstrap_ci(
                paired_rows,
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
            "  mean:",
            f"{mean(values):+.6f}",
        )

        print(
            "  median:",
            f"{median(values):+.6f}",
        )

        print(
            "  95% class-clustered "
            "bootstrap CI:",
            f"[{ci_low:+.6f}, "
            f"{ci_high:+.6f}]",
        )


if __name__ == "__main__":

    analyze_pair(
        name="Background Blur",
        baseline_path=(
            RESULTS_DIR
            / "finetuned_lambda_0p0_background_blur.csv"
        ),
        method_path=(
            RESULTS_DIR
            / "finetuned_lambda_0p01_background_blur.csv"
        ),
    )

    analyze_pair(
        name="Background Mask",
        baseline_path=(
            RESULTS_DIR
            / "finetuned_lambda_0p0_background_mask.csv"
        ),
        method_path=(
            RESULTS_DIR
            / "finetuned_lambda_0p01_background_mask.csv"
        ),
    )