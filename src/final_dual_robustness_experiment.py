
import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image

import final_robustness_experiment as base


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one checkpoint on the frozen "
            "5394-image final cohort using both "
            "background blur and background mask "
            "while reusing the original-image forward pass."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--blur-output",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--mask-output",
        type=str,
        required=True,
    )

    return parser.parse_args()


def build_result(
    record,
    perturbation_name,
    original_pred_index,
    original_pred_class_id,
    original_logits,
    original_attention,
    perturbed_logits,
    perturbed_attention,
    changed_inside,
):
    perturbed_pred_index = (
        perturbed_logits.argmax().item()
    )

    perturbed_pred_class_id = (
        perturbed_pred_index + 1
    )

    prediction_unchanged = (
        original_pred_index
        == perturbed_pred_index
    )

    # Same protocol as final_robustness_experiment:
    # use ORIGINAL predicted class query for both.
    fixed_query_index = (
        original_pred_index
    )

    original_query_attention = (
        original_attention[
            fixed_query_index
        ]
    )

    perturbed_query_attention = (
        perturbed_attention[
            fixed_query_index
        ]
    )

    cosine = base.cosine_similarity(
        original_query_attention,
        perturbed_query_attention,
    )

    pearson = base.pearson_correlation(
        original_query_attention,
        perturbed_query_attention,
    )

    top20_iou = base.topk_iou(
        original_query_attention,
        perturbed_query_attention,
        fraction=base.TOP_ATTENTION_FRACTION,
    )

    original_margin = base.class_margin(
        original_logits,
        fixed_query_index,
    )

    perturbed_margin = base.class_margin(
        perturbed_logits,
        fixed_query_index,
    )

    margin_change = (
        perturbed_margin
        - original_margin
    )

    category = base.classify_abcd(
        prediction_unchanged,
        cosine,
        top20_iou,
    )

    return {
        "image_id":
            record["image_id"],

        "class_id":
            record["class_id"],

        "class_name":
            record["class_name"],

        "perturbation":
            perturbation_name,

        "original_pred_class_id":
            original_pred_class_id,

        "perturbed_pred_class_id":
            perturbed_pred_class_id,

        "original_correct":
            (
                original_pred_class_id
                ==
                int(record["class_id"])
            ),

        "perturbed_correct":
            (
                perturbed_pred_class_id
                ==
                int(record["class_id"])
            ),

        "prediction_unchanged":
            prediction_unchanged,

        "fixed_query_index":
            fixed_query_index,

        "original_margin":
            original_margin,

        "perturbed_margin":
            perturbed_margin,

        "margin_change":
            margin_change,

        "attention_cosine":
            cosine,

        "attention_pearson":
            pearson,

        "attention_top20_iou":
            top20_iou,

        "changed_bbox_pixel_values":
            changed_inside,

        "category":
            category,
    }


def evaluate_dual_sample(
    model,
    record,
    transform,
    device,
):
    # Original image
    original = Image.open(
        record["image_path"]
    ).convert("RGB")

    # Same perturbations as existing evaluator
    blur = base.background_blur(
        original,
        record["bounding_box"],
        radius=base.BACKGROUND_BLUR_RADIUS,
    )

    mask = base.background_mask(
        original,
        record["bounding_box"],
        fill_color=(128, 128, 128),
    )

    # BBox integrity
    blur_changed_inside = (
        base.count_changed_bbox_pixels(
            original,
            blur,
            record["bounding_box"],
        )
    )

    mask_changed_inside = (
        base.count_changed_bbox_pixels(
            original,
            mask,
            record["bounding_box"],
        )
    )

    assert blur_changed_inside == 0
    assert mask_changed_inside == 0

    # Same deterministic INTR test preprocessing
    original_tensor, _ = transform(
        original,
        None,
    )

    blur_tensor, _ = transform(
        blur,
        None,
    )

    mask_tensor, _ = transform(
        mask,
        None,
    )

    # ORIGINAL FORWARD: only once
    (
        original_logits,
        original_attention,
        _,
    ) = base.run_intr(
        model,
        original_tensor,
        device,
    )

    original_pred_index = (
        original_logits.argmax().item()
    )

    original_pred_class_id = (
        original_pred_index + 1
    )

    # Blur forward
    (
        blur_logits,
        blur_attention,
        _,
    ) = base.run_intr(
        model,
        blur_tensor,
        device,
    )

    # Mask forward
    (
        mask_logits,
        mask_attention,
        _,
    ) = base.run_intr(
        model,
        mask_tensor,
        device,
    )

    blur_result = build_result(
        record=record,
        perturbation_name="background_blur",
        original_pred_index=original_pred_index,
        original_pred_class_id=original_pred_class_id,
        original_logits=original_logits,
        original_attention=original_attention,
        perturbed_logits=blur_logits,
        perturbed_attention=blur_attention,
        changed_inside=blur_changed_inside,
    )

    mask_result = build_result(
        record=record,
        perturbation_name="background_mask",
        original_pred_index=original_pred_index,
        original_pred_class_id=original_pred_class_id,
        original_logits=original_logits,
        original_attention=original_attention,
        perturbed_logits=mask_logits,
        perturbed_attention=mask_attention,
        changed_inside=mask_changed_inside,
    )

    return blur_result, mask_result


def save_results(
    results,
    output_path,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)


def print_summary(
    results,
    name,
):
    total = len(results)

    counts = Counter(
        row["category"]
        for row in results
    )

    print()
    print(
        f"=== {name} Final Category Summary ==="
    )

    for category in [
        "A", "B", "C", "D"
    ]:
        count = counts[category]

        print(
            f"{category}: "
            f"{count}/{total} "
            f"({count / total:.1%})"
        )

    prediction_stable = (
        counts["A"]
        + counts["B"]
    )

    attention_unstable = (
        counts["B"]
        + counts["D"]
    )

    print()

    print(
        "Prediction stable:",
        f"{prediction_stable}/{total}",
        f"({prediction_stable / total:.1%})",
    )

    print(
        "Attention unstable:",
        f"{attention_unstable}/{total}",
        f"({attention_unstable / total:.1%})",
    )

    if prediction_stable > 0:
        print(
            "B among prediction-stable:",
            f"{counts['B']}/"
            f"{prediction_stable}",
            f"({counts['B'] / prediction_stable:.1%})",
        )

    clean_correct = sum(
        row["original_correct"]
        for row in results
    )

    perturbed_correct = sum(
        row["perturbed_correct"]
        for row in results
    )

    print()
    print(
        "=== Clean / Perturbed Correctness ==="
    )

    print(
        "Clean correct on frozen final cohort:",
        f"{clean_correct}/{total}",
        f"({clean_correct / total:.1%})",
    )

    print(
        "Perturbed correct:",
        f"{perturbed_correct}/{total}",
        f"({perturbed_correct / total:.1%})",
    )

    all_bbox_preserved = all(
        row[
            "changed_bbox_pixel_values"
        ] == 0
        for row in results
    )

    print(
        "All bbox interiors preserved:",
        all_bbox_preserved,
    )


if __name__ == "__main__":

    cli_args = parse_args()

    args = base.get_intr_config()

    import torch

    if torch.cuda.is_available():
        args.device = "cuda:0"
    else:
        args.device = "cpu"

    print("Evaluation device:", args.device)

    checkpoint_path = Path(
        cli_args.checkpoint
    )

    blur_output = Path(
        cli_args.blur_output
    )

    mask_output = Path(
        cli_args.mask_output
    )

    assert checkpoint_path.exists(), (
        f"Checkpoint not found: "
        f"{checkpoint_path}"
    )

    print("Checkpoint:", checkpoint_path)
    print("Blur output:", blur_output)
    print("Mask output:", mask_output)

    print("Loading INTR...")

    model = base.load_intr_model(
        args,
        checkpoint_path,
    )

    transform = (
        base.make_intr_test_transform()
    )

    dataset = base.CUBDataset(
        root=base.CUB_ROOT,
        split="test",
        transform=None,
    )

    dataset_lookup = (
        base.build_dataset_lookup(
            dataset
        )
    )

    final_rows = base.load_final_set(
        base.FINAL_SET_PATH
    )

    final_image_ids = [
        int(row["image_id"])
        for row in final_rows
    ]

    print()
    print("=== Frozen Final Set Check ===")
    print("Final rows:", len(final_rows))
    print(
        "Unique image IDs:",
        len(set(final_image_ids)),
    )

    assert len(final_rows) == 5394
    assert len(set(final_image_ids)) == 5394

    missing_ids = [
        image_id
        for image_id in final_image_ids
        if image_id not in dataset_lookup
    ]

    print(
        "Missing image IDs:",
        missing_ids,
    )

    assert not missing_ids

    blur_results = []
    mask_results = []

    total = len(final_rows)

    print()
    print(
        "=== Running Dual Final Evaluation ==="
    )

    for index, final_row in enumerate(
        final_rows,
        start=1,
    ):
        image_id = int(
            final_row["image_id"]
        )

        record = dataset_lookup[
            image_id
        ]

        assert int(
            final_row["class_id"]
        ) == int(
            record["class_id"]
        )

        (
            blur_result,
            mask_result,
        ) = evaluate_dual_sample(
            model=model,
            record=record,
            transform=transform,
            device=args.device,
        )

        checkpoint_name = (
            checkpoint_path.name
        )

        blur_result[
            "checkpoint_name"
        ] = checkpoint_name

        mask_result[
            "checkpoint_name"
        ] = checkpoint_name

        blur_results.append(
            blur_result
        )

        mask_results.append(
            mask_result
        )

        if (
            index == 1
            or index % 100 == 0
            or index == total
        ):
            print(
                f"{index:4d}/{total} | "
                f"Image {image_id:5d} | "
                f"Orig "
                f"{blur_result['original_pred_class_id']:3d} | "
                f"Blur "
                f"{blur_result['perturbed_pred_class_id']:3d} "
                f"({blur_result['category']}) | "
                f"Mask "
                f"{mask_result['perturbed_pred_class_id']:3d} "
                f"({mask_result['category']})"
            )

    save_results(
        blur_results,
        blur_output,
    )

    save_results(
        mask_results,
        mask_output,
    )

    print()
    print(
        "Saved blur results to:",
        blur_output,
    )

    print(
        "Saved mask results to:",
        mask_output,
    )

    print_summary(
        blur_results,
        "BLUR",
    )

    print_summary(
        mask_results,
        "MASK",
    )
