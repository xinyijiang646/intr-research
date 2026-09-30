import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def load_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


if __name__ == "__main__":

    seed42 = load_csv(
        RESULTS_DIR
        / "background_blur_50_seed42.csv"
    )

    seed43 = load_csv(
        RESULTS_DIR
        / "background_blur_50_seed43.csv"
    )

    images42 = {
        int(row["image_id"])
        for row in seed42
    }

    images43 = {
        int(row["image_id"])
        for row in seed43
    }

    classes42 = {
        int(row["class_id"])
        for row in seed42
    }

    classes43 = {
        int(row["class_id"])
        for row in seed43
    }

    overlapping_images = (
        images42 & images43
    )

    overlapping_classes = (
        classes42 & classes43
    )

    unique_images = (
        images42 | images43
    )

    print("=== Seed Comparison ===")

    print(
        "Seed 42 images:",
        len(images42),
    )

    print(
        "Seed 43 images:",
        len(images43),
    )

    print(
        "Overlapping images:",
        len(overlapping_images),
    )

    print(
        "Unique images across both:",
        len(unique_images),
    )

    print(
        "Overlapping classes:",
        len(overlapping_classes),
    )

    print()
    print(
        "Overlapping image IDs:",
        sorted(overlapping_images),
    )