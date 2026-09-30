import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

BASELINE_PATH = (
    RESULTS_DIR
    / "full_test_baseline.csv"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "formal_400_samples.csv"
)

FORMAL_SAMPLE_SEED = 42
SAMPLES_PER_CLASS = 2


def load_baseline_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def str_to_bool(value):
    return value.strip().lower() == "true"


if __name__ == "__main__":

    rows = load_baseline_csv(
        BASELINE_PATH
    )

    print("=== Loading Full-Test Baseline ===")
    print(
        "Total baseline rows:",
        len(rows),
    )

    # Keep only baseline-correct images
    correct_rows = [
        row
        for row in rows
        if str_to_bool(
            row["top1_correct"]
        )
    ]

    print(
        "Baseline-correct rows:",
        len(correct_rows),
    )

    # Group by class
    by_class = defaultdict(list)

    for row in correct_rows:
        class_id = int(
            row["class_id"]
        )

        by_class[class_id].append(
            row
        )

    print(
        "Classes with correct images:",
        len(by_class),
    )

    # Verify formal sampling is possible
    insufficient_classes = []

    for class_id in range(1, 201):

        num_correct = len(
            by_class[class_id]
        )

        if (
            num_correct
            < SAMPLES_PER_CLASS
        ):
            insufficient_classes.append(
                (
                    class_id,
                    num_correct,
                )
            )

    if insufficient_classes:

        print()
        print(
            "Classes with insufficient "
            "baseline-correct images:"
        )

        for (
            class_id,
            num_correct,
        ) in insufficient_classes:

            print(
                f"Class {class_id:3d}: "
                f"{num_correct} correct"
            )

        raise RuntimeError(
            "Cannot construct strict "
            "2-per-class formal set."
        )

    print()
    print(
        "All 200 classes have at least "
        f"{SAMPLES_PER_CLASS} "
        "baseline-correct images."
    )

    # Reproducible class-balanced sampling
    rng = random.Random(
        FORMAL_SAMPLE_SEED
    )

    selected_rows = []

    for class_id in range(1, 201):

        candidates = list(
            by_class[class_id]
        )

        # Sort before shuffle so the result is
        # reproducible independent of CSV row order.
        candidates.sort(
            key=lambda row: int(
                row["image_id"]
            )
        )

        rng.shuffle(
            candidates
        )

        chosen = candidates[
            :SAMPLES_PER_CLASS
        ]

        for sample_number, row in enumerate(
            chosen,
            start=1,
        ):

            selected_rows.append(
                {
                    "image_id":
                        int(
                            row["image_id"]
                        ),

                    "class_id":
                        int(
                            row["class_id"]
                        ),

                    "class_name":
                        row["class_name"],

                    "predicted_class_id":
                        int(
                            row[
                                "predicted_class_id"
                            ]
                        ),

                    "baseline_top1_correct":
                        str_to_bool(
                            row[
                                "top1_correct"
                            ]
                        ),

                    "selection_seed":
                        FORMAL_SAMPLE_SEED,

                    "within_class_sample_number":
                        sample_number,
                }
            )

    # Validation
    total_selected = len(
        selected_rows
    )

    image_ids = [
        row["image_id"]
        for row in selected_rows
    ]

    class_ids = [
        row["class_id"]
        for row in selected_rows
    ]

    unique_image_ids = set(
        image_ids
    )

    unique_class_ids = set(
        class_ids
    )

    class_counts = Counter(
        class_ids
    )

    all_two_per_class = all(
        count == SAMPLES_PER_CLASS
        for count
        in class_counts.values()
    )

    all_baseline_correct = all(
        row["baseline_top1_correct"]
        for row in selected_rows
    )

    prediction_matches_class = all(
        (
            row["predicted_class_id"]
            == row["class_id"]
        )
        for row in selected_rows
    )

    print()
    print("=== Formal 400 Set Validation ===")

    duplicate_image_ids = [
        image_id
        for image_id, count
        in Counter(image_ids).items()
        if count > 1
    ]

    print(
        "Duplicate image IDs:",
        duplicate_image_ids,
    )

    print(
        "Total selected:",
        total_selected,
    )

    print(
        "Unique images:",
        len(
            unique_image_ids
        ),
    )

    print(
        "Unique classes:",
        len(
            unique_class_ids
        ),
    )

    print(
        "Exactly 2 images per class:",
        all_two_per_class,
    )

    print(
        "All baseline correct:",
        all_baseline_correct,
    )

    print(
        "Predicted class matches true class:",
        prediction_matches_class,
    )

    # Strict assertions
    assert total_selected == 400

    assert (
        len(unique_image_ids)
        == 400
    )

    assert (
        len(unique_class_ids)
        == 200
    )

    assert all_two_per_class

    assert all_baseline_correct

    assert prediction_matches_class

    # Save frozen formal cohort
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        fieldnames = [
            "image_id",
            "class_id",
            "class_name",
            "predicted_class_id",
            "baseline_top1_correct",
            "selection_seed",
            "within_class_sample_number",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            selected_rows
        )

    print()
    print(
        "Saved frozen formal set to:"
    )

    print(
        OUTPUT_PATH
    )

    # Show a few examples
    print()
    print("=== First 10 Selected Samples ===")

    for row in selected_rows[:10]:

        print(
            f"Class "
            f"{row['class_id']:3d} | "
            f"Sample "
            f"{row['within_class_sample_number']} | "
            f"Image "
            f"{row['image_id']:5d} | "
            f"{row['class_name']}"
        )
