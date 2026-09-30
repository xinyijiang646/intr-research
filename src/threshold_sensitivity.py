import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "background_blur_50_seed42.csv"
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


def load_results(csv_path):
    results = []

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            results.append({
                "image_id":
                    int(row["image_id"]),

                "prediction_unchanged":
                    row["prediction_unchanged"].lower()
                    == "true",

                "cosine":
                    float(row["attention_cosine"]),

                "iou":
                    float(row["attention_top20_iou"]),
            })

    return results


def classify(
    prediction_unchanged,
    cosine,
    iou,
    cosine_threshold,
    iou_threshold,
):
    attention_stable = (
        cosine >= cosine_threshold
        and
        iou >= iou_threshold
    )

    if prediction_unchanged and attention_stable:
        return "A"

    if prediction_unchanged and not attention_stable:
        return "B"

    if not prediction_unchanged and attention_stable:
        return "C"

    return "D"


if __name__ == "__main__":

    results = load_results(CSV_PATH)

    print("=== Threshold Sensitivity Analysis ===")
    print("Samples:", len(results))

    print()
    print(
        f"{'Cos':>6} "
        f"{'IoU':>6} "
        f"{'A':>5} "
        f"{'B':>5} "
        f"{'C':>5} "
        f"{'D':>5} "
        f"{'B/all':>8} "
        f"{'B/stable-pred':>14}"
    )

    print("-" * 65)

    for cosine_threshold in COSINE_THRESHOLDS:

        for iou_threshold in IOU_THRESHOLDS:

            counts = {
                "A": 0,
                "B": 0,
                "C": 0,
                "D": 0,
            }

            for row in results:

                category = classify(
                    prediction_unchanged=
                        row["prediction_unchanged"],

                    cosine=
                        row["cosine"],

                    iou=
                        row["iou"],

                    cosine_threshold=
                        cosine_threshold,

                    iou_threshold=
                        iou_threshold,
                )

                counts[category] += 1

            total = len(results)

            prediction_stable_total = (
                counts["A"] + counts["B"]
            )

            b_rate_all = (
                counts["B"] / total
            )

            b_rate_stable = (
                counts["B"]
                / prediction_stable_total
            )

            print(
                f"{cosine_threshold:6.2f} "
                f"{iou_threshold:6.2f} "
                f"{counts['A']:5d} "
                f"{counts['B']:5d} "
                f"{counts['C']:5d} "
                f"{counts['D']:5d} "
                f"{b_rate_all:8.1%} "
                f"{b_rate_stable:14.1%}"
            )