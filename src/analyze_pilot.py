import csv
import statistics
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "background_blur_50_seed42.csv"
)


def read_results(csv_path):
    rows = []

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            # Convert numeric fields back to numbers
            row["image_id"] = int(row["image_id"])
            row["class_id"] = int(row["class_id"])

            row["original_pred_class_id"] = int(
                row["original_pred_class_id"]
            )

            row["perturbed_pred_class_id"] = int(
                row["perturbed_pred_class_id"]
            )

            row["original_margin"] = float(
                row["original_margin"]
            )

            row["perturbed_margin"] = float(
                row["perturbed_margin"]
            )

            row["margin_change"] = float(
                row["margin_change"]
            )

            row["attention_cosine"] = float(
                row["attention_cosine"]
            )

            row["attention_pearson"] = float(
                row["attention_pearson"]
            )

            row["attention_top20_iou"] = float(
                row["attention_top20_iou"]
            )

            row["changed_bbox_pixel_values"] = int(
                row["changed_bbox_pixel_values"]
            )

            rows.append(row)

    return rows


def mean(values):
    return sum(values) / len(values)


if __name__ == "__main__":
    results = read_results(CSV_PATH)

    print("=== Pilot Summary ===")
    print("Number of samples:", len(results))

    # Sanity checks
    all_original_correct = all(
        row["original_pred_class_id"]
        == row["class_id"]
        for row in results
    )

    all_bbox_preserved = all(
        row["changed_bbox_pixel_values"] == 0
        for row in results
    )

    print()
    print("=== Sanity Checks ===")
    print(
        "All original predictions correct:",
        all_original_correct,
    )
    print(
        "All bbox interiors pixel-identical:",
        all_bbox_preserved,
    )

    # A/B/C/D counts
    counts = Counter(
        row["category"]
        for row in results
    )

    print()
    print("=== Category Counts ===")

    for category in ["A", "B", "C", "D"]:
        count = counts.get(category, 0)

        print(
            f"{category}: "
            f"{count}/{len(results)} "
            f"({count / len(results):.1%})"
        )

    # Overall metric means
    mean_cosine = mean([
        row["attention_cosine"]
        for row in results
    ])

    mean_pearson = mean([
        row["attention_pearson"]
        for row in results
    ])

    mean_iou = mean([
        row["attention_top20_iou"]
        for row in results
    ])

    mean_margin_change = mean([
        row["margin_change"]
        for row in results
    ])

    print()
    print("=== Mean Metrics ===")
    print(
        f"Mean cosine:       {mean_cosine:.4f}"
    )
    print(
        f"Mean Pearson:      {mean_pearson:.4f}"
    )
    print(
        f"Mean top-20% IoU:  {mean_iou:.4f}"
    )
    print(
        f"Mean margin change:{mean_margin_change:.4f}"
    )

    # B cases
    b_cases = [
        row
        for row in results
        if row["category"] == "B"
    ]

    print()
    print("=== Category B Cases ===")

    if not b_cases:
        print("No B cases found.")
    else:
        for row in b_cases:
            print(
                f"Image {row['image_id']:4d} | "
                f"Class {row['class_id']:3d} | "
                f"Cos {row['attention_cosine']:.3f} | "
                f"Pearson {row['attention_pearson']:.3f} | "
                f"IoU {row['attention_top20_iou']:.3f} | "
                f"Margin Δ {row['margin_change']:.3f}"
            )

    # Prediction change rate
    prediction_changed = [
        row
        for row in results
        if row["original_pred_class_id"]
        != row["perturbed_pred_class_id"]
    ]

    print()
    print("=== Prediction Stability ===")
    print(
        "Prediction change rate:",
        f"{len(prediction_changed)}/{len(results)} "
        f"({len(prediction_changed) / len(results):.1%})"
    )

    # Mean + median metrics
    cosines = [
        row["attention_cosine"]
        for row in results
    ]

    pearsons = [
        row["attention_pearson"]
        for row in results
    ]

    ious = [
        row["attention_top20_iou"]
        for row in results
    ]

    margin_changes = [
        row["margin_change"]
        for row in results
    ]

    print()
    print("=== Metric Summary ===")

    print(
        f"Cosine: "
        f"mean={statistics.mean(cosines):.4f}, "
        f"median={statistics.median(cosines):.4f}, "
        f"std={statistics.stdev(cosines):.4f}"
    )

    print(
        f"Pearson: "
        f"mean={statistics.mean(pearsons):.4f}, "
        f"median={statistics.median(pearsons):.4f}, "
        f"std={statistics.stdev(pearsons):.4f}"
    )

    print(
        f"Top-20% IoU: "
        f"mean={statistics.mean(ious):.4f}, "
        f"median={statistics.median(ious):.4f}, "
        f"std={statistics.stdev(ious):.4f}"
    )

    print(
        f"Margin change: "
        f"mean={statistics.mean(margin_changes):.4f}, "
        f"median={statistics.median(margin_changes):.4f}, "
        f"std={statistics.stdev(margin_changes):.4f}"
    )

    # Compare A and B groups
    for category in ["A", "B"]:

        group = [
            row
            for row in results
            if row["category"] == category
        ]

        if not group:
            continue

        print()
        print(f"=== Category {category} Summary ===")
        print("Count:", len(group))

        print(
            "Mean cosine:",
            f"{statistics.mean([
                row['attention_cosine']
                for row in group
            ]):.4f}"
        )

        print(
            "Mean Pearson:",
            f"{statistics.mean([
                row['attention_pearson']
                for row in group
            ]):.4f}"
        )

        print(
            "Mean IoU:",
            f"{statistics.mean([
                row['attention_top20_iou']
                for row in group
            ]):.4f}"
        )

        print(
            "Mean margin change:",
            f"{statistics.mean([
                row['margin_change']
                for row in group
            ]):.4f}"
        )

    # --------------------------------------------------
    # Extreme cases
    # --------------------------------------------------

    print()
    print("=== Lowest Cosine Cases ===")

    for row in sorted(
        results,
        key=lambda r: r["attention_cosine"],
    )[:5]:
        print(
            f"Image {row['image_id']:4d} | "
            f"Class {row['class_id']:3d} | "
            f"Category {row['category']} | "
            f"Cos {row['attention_cosine']:.3f} | "
            f"IoU {row['attention_top20_iou']:.3f} | "
            f"Margin Δ {row['margin_change']:.3f}"
        )

    print()
    print("=== Lowest IoU Cases ===")

    for row in sorted(
        results,
        key=lambda r: r["attention_top20_iou"],
    )[:5]:
        print(
            f"Image {row['image_id']:4d} | "
            f"Class {row['class_id']:3d} | "
            f"Category {row['category']} | "
            f"Cos {row['attention_cosine']:.3f} | "
            f"IoU {row['attention_top20_iou']:.3f} | "
            f"Margin Δ {row['margin_change']:.3f}"
        )

    print()
    print("=== Largest Margin Drops ===")

    for row in sorted(
        results,
        key=lambda r: r["margin_change"],
    )[:5]:
        print(
            f"Image {row['image_id']:4d} | "
            f"Class {row['class_id']:3d} | "
            f"Category {row['category']} | "
            f"Margin Δ {row['margin_change']:.3f} | "
            f"Cos {row['attention_cosine']:.3f} | "
            f"IoU {row['attention_top20_iou']:.3f}"
        )