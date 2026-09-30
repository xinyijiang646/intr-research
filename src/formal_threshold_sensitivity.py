import csv
from collections import Counter
from pathlib import Path


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


COSINE_THRESHOLDS = [
    0.85,
    0.90,
    0.95,
]

IOU_THRESHOLDS = [
    0.40,
    0.50,
    0.60,
]


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


def classify_abcd(
    prediction_unchanged,
    cosine,
    top20_iou,
    cosine_threshold,
    iou_threshold,
):
    attention_stable = (
        cosine >= cosine_threshold
        and
        top20_iou >= iou_threshold
    )

    if prediction_unchanged:
        if attention_stable:
            return "A"
        return "B"

    if attention_stable:
        return "C"

    return "D"


def evaluate_threshold_pair(
    rows,
    cosine_threshold,
    iou_threshold,
):
    categories = []

    for row in rows:

        prediction_unchanged = (
            str_to_bool(
                row[
                    "prediction_unchanged"
                ]
            )
        )

        cosine = float(
            row[
                "attention_cosine"
            ]
        )

        top20_iou = float(
            row[
                "attention_top20_iou"
            ]
        )

        category = classify_abcd(
            prediction_unchanged,
            cosine,
            top20_iou,
            cosine_threshold,
            iou_threshold,
        )

        categories.append(
            category
        )

    counts = Counter(
        categories
    )

    total = len(
        categories
    )

    stable_prediction = (
        counts["A"]
        + counts["B"]
    )

    if stable_prediction > 0:
        b_among_stable = (
            counts["B"]
            / stable_prediction
        )
    else:
        b_among_stable = 0.0

    return {
        "A": counts["A"],
        "B": counts["B"],
        "C": counts["C"],
        "D": counts["D"],
        "B_all":
            counts["B"] / total,
        "B_stable":
            b_among_stable,
    }


def print_table(
    name,
    rows,
):
    print()
    print(
        f"=== {name} Threshold Sensitivity ==="
    )

    print(
        "Cos  IoU   "
        "A    B    C    D    "
        "B/all   B/stable"
    )

    for cosine_threshold in (
        COSINE_THRESHOLDS
    ):
        for iou_threshold in (
            IOU_THRESHOLDS
        ):

            result = (
                evaluate_threshold_pair(
                    rows,
                    cosine_threshold,
                    iou_threshold,
                )
            )

            print(
                f"{cosine_threshold:.2f} "
                f"{iou_threshold:.2f}  "
                f"{result['A']:3d} "
                f"{result['B']:3d} "
                f"{result['C']:3d} "
                f"{result['D']:3d}   "
                f"{result['B_all']:.1%}   "
                f"{result['B_stable']:.1%}"
            )


if __name__ == "__main__":

    blur_rows = load_csv(
        BLUR_PATH
    )

    mask_rows = load_csv(
        MASK_PATH
    )

    print(
        "=== Formal Threshold Sensitivity ==="
    )

    print(
        "Blur rows:",
        len(blur_rows),
    )

    print(
        "Mask rows:",
        len(mask_rows),
    )

    assert len(blur_rows) == 400
    assert len(mask_rows) == 400

    print_table(
        "Background Blur",
        blur_rows,
    )

    print_table(
        "Background Mask",
        mask_rows,
    )

    print()
    print(
        "Primary pre-specified threshold:"
    )

    print(
        "Cosine >= 0.90 and "
        "Top-20% IoU >= 0.50"
    )