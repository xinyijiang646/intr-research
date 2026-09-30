import csv
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from config import get_intr_config
from model import load_intr_model

from dataset import (
    CUBDataset,
    CUB_ROOT,
    make_intr_test_transform,
)

from robustness import run_intr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_PATH = (
    RESULTS_DIR
    / "full_test_baseline.csv"
)


def evaluate_full_test(
    model,
    dataset,
    transform,
    device,
):
    """
    Run baseline INTR inference on every image
    in the CUB test split.

    No perturbation is applied.
    """

    results = []

    total = len(dataset)

    top1_correct = 0
    top5_correct = 0

    print()
    print("=== Full CUB Test Baseline Scan ===")
    print("Total test images:", total)
    print()

    for index, record in enumerate(
        dataset.records
    ):
        image = Image.open(
            record["image_path"]
        ).convert("RGB")

        image_tensor, _ = transform(
            image,
            None,
        )

        logits, _, _ = run_intr(
            model,
            image_tensor,
            device,
        )

        true_class_id = record["class_id"]
        true_index = true_class_id - 1

        # ------------------------------
        # Top-1
        # ------------------------------

        predicted_index = (
            logits.argmax().item()
        )

        predicted_class_id = (
            predicted_index + 1
        )

        is_top1_correct = (
            predicted_index == true_index
        )

        if is_top1_correct:
            top1_correct += 1

        # ------------------------------
        # Top-5
        # ------------------------------

        top5_indices = torch.topk(
            logits,
            k=5,
        ).indices.tolist()

        is_top5_correct = (
            true_index in top5_indices
        )

        if is_top5_correct:
            top5_correct += 1

        top5_class_ids = [
            class_index + 1
            for class_index in top5_indices
        ]

        # ------------------------------
        # Store result
        # ------------------------------

        results.append(
            {
                "image_id":
                    record["image_id"],

                "class_id":
                    true_class_id,

                "class_name":
                    record["class_name"],

                "predicted_class_id":
                    predicted_class_id,

                "top1_correct":
                    is_top1_correct,

                "top5_correct":
                    is_top5_correct,

                "top5_class_ids":
                    " ".join(
                        str(class_id)
                        for class_id
                        in top5_class_ids
                    ),
            }
        )

        # ------------------------------
        # Progress
        # ------------------------------

        current = index + 1

        if (
            current % 100 == 0
            or current == total
        ):
            running_top1 = (
                top1_correct / current
            )

            running_top5 = (
                top5_correct / current
            )

            print(
                f"{current:4d}/{total} | "
                f"Top-1 {running_top1:.2%} | "
                f"Top-5 {running_top5:.2%}"
            )

    return (
        results,
        top1_correct,
        top5_correct,
    )


def analyze_class_availability(results):
    """
    Count test images and baseline-correct images
    separately for each CUB class.
    """

    total_by_class = defaultdict(int)
    correct_by_class = defaultdict(int)

    class_names = {}

    for row in results:
        class_id = row["class_id"]

        total_by_class[class_id] += 1

        class_names[class_id] = (
            row["class_name"]
        )

        if row["top1_correct"]:
            correct_by_class[class_id] += 1

    print()
    print("=== Per-Class Baseline Availability ===")

    for class_id in sorted(
        total_by_class.keys()
    ):
        total = total_by_class[class_id]
        correct = correct_by_class[class_id]

        print(
            f"Class {class_id:3d} | "
            f"Test {total:2d} | "
            f"Top-1 correct {correct:2d} | "
            f"{class_names[class_id]}"
        )

    # ----------------------------------
    # Availability thresholds
    # ----------------------------------

    classes = sorted(
        total_by_class.keys()
    )

    at_least_1 = [
        class_id
        for class_id in classes
        if correct_by_class[class_id] >= 1
    ]

    at_least_2 = [
        class_id
        for class_id in classes
        if correct_by_class[class_id] >= 2
    ]

    at_least_3 = [
        class_id
        for class_id in classes
        if correct_by_class[class_id] >= 3
    ]

    zero_correct = [
        class_id
        for class_id in classes
        if correct_by_class[class_id] == 0
    ]

    exactly_1 = [
        class_id
        for class_id in classes
        if correct_by_class[class_id] == 1
    ]

    print()
    print("=== Formal-Set Availability Summary ===")

    print(
        "Total classes:",
        len(classes),
    )

    print(
        "Classes with >= 1 correct:",
        len(at_least_1),
    )

    print(
        "Classes with >= 2 correct:",
        len(at_least_2),
    )

    print(
        "Classes with >= 3 correct:",
        len(at_least_3),
    )

    print(
        "Classes with 0 correct:",
        len(zero_correct),
    )

    print(
        "Classes with exactly 1 correct:",
        len(exactly_1),
    )

    print()
    print(
        "Classes with 0 correct:",
        zero_correct,
    )

    print(
        "Classes with exactly 1 correct:",
        exactly_1,
    )

    can_build_400 = (
        len(at_least_2) == 200
    )

    print()
    print(
        "Can construct strict "
        "2-per-class 400-image set:",
        can_build_400,
    )

    if can_build_400:
        print(
            "Formal-set capacity: "
            "200 classes x 2 correct images "
            "= 400 images"
        )

    else:
        maximum_strict_two_per_class = (
            len(at_least_2) * 2
        )

        print(
            "Maximum strict 2-per-class "
            "set using classes with >=2 correct:",
            maximum_strict_two_per_class,
        )

    return {
        "at_least_1": at_least_1,
        "at_least_2": at_least_2,
        "at_least_3": at_least_3,
        "zero_correct": zero_correct,
        "exactly_1": exactly_1,
        "can_build_400": can_build_400,
    }


if __name__ == "__main__":

    args = get_intr_config()

    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "intr_checkpoint_cub_detr_r50.pth"
    )

    print("Loading INTR...")

    model = load_intr_model(
        args,
        checkpoint_path,
    )

    transform = make_intr_test_transform()

    # Raw CUB test dataset.
    dataset = CUBDataset(
        root=CUB_ROOT,
        split="test",
        transform=None,
    )

    # ----------------------------------
    # Full-test baseline inference
    # ----------------------------------

    (
        results,
        top1_correct,
        top5_correct,
    ) = evaluate_full_test(
        model=model,
        dataset=dataset,
        transform=transform,
        device=args.device,
    )

    total = len(results)

    top1_accuracy = (
        top1_correct / total
    )

    top5_accuracy = (
        top5_correct / total
    )

    print()
    print("=== Full-Test Baseline Results ===")

    print(
        "Total test images:",
        total,
    )

    print(
        "Top-1 correct:",
        f"{top1_correct}/{total}",
    )

    print(
        "Top-1 accuracy:",
        f"{top1_accuracy:.4%}",
    )

    print(
        "Top-5 correct:",
        f"{top5_correct}/{total}",
    )

    print(
        "Top-5 accuracy:",
        f"{top5_accuracy:.4%}",
    )

    # ----------------------------------
    # Save baseline results
    # ----------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("Saved full-test baseline to:")
    print(OUTPUT_PATH)

    # ----------------------------------
    # Per-class availability analysis
    # ----------------------------------

    analyze_class_availability(
        results
    )