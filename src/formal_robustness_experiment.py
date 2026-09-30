import csv
import argparse
from collections import Counter
from pathlib import Path

from PIL import Image

from config import (
    get_intr_config,
    BACKGROUND_BLUR_RADIUS,
    ATTENTION_COSINE_THRESHOLD,
    ATTENTION_TOP20_IOU_THRESHOLD,
    TOP_ATTENTION_FRACTION,
)

from model import load_intr_model

from dataset import (
    CUBDataset,
    CUB_ROOT,
    make_intr_test_transform,
)

from perturbations import (
    background_blur,
    background_mask,
    count_changed_bbox_pixels,
)

from robustness import (
    run_intr,
    cosine_similarity,
    pearson_correlation,
    topk_iou,
    class_margin,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

FORMAL_SET_PATH = (
    RESULTS_DIR
    / "formal_400_samples.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a checkpoint on the frozen "
            "formal 400-image robustness cohort."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )

    parser.add_argument(
        "--perturbation",
        type=str,
        required=True,
        choices=[
            "background_blur",
            "background_mask",
        ],
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV path.",
    )

    return parser.parse_args()


def load_formal_set(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def classify_abcd(
    prediction_unchanged,
    cosine,
    top20_iou,
):
    attention_stable = (
        cosine
        >= ATTENTION_COSINE_THRESHOLD
        and
        top20_iou
        >= ATTENTION_TOP20_IOU_THRESHOLD
    )

    if (
        prediction_unchanged
        and attention_stable
    ):
        return "A"

    if (
        prediction_unchanged
        and not attention_stable
    ):
        return "B"

    if (
        not prediction_unchanged
        and attention_stable
    ):
        return "C"

    return "D"


def apply_perturbation(
    image,
    bounding_box,
    perturbation_name,
):
    if perturbation_name == "background_blur":
        return background_blur(
            image,
            bounding_box,
            radius=BACKGROUND_BLUR_RADIUS,
        )

    if perturbation_name == "background_mask":
        return background_mask(
            image,
            bounding_box,
            fill_color=(128, 128, 128),
        )

    raise ValueError(
        f"Unknown perturbation: "
        f"{perturbation_name}"
    )


def build_dataset_lookup(dataset):
    """
    Map image_id -> dataset record.
    """

    lookup = {}

    for record in dataset.records:
        image_id = int(
            record["image_id"]
        )

        lookup[image_id] = record

    return lookup


def evaluate_sample(
    model,
    record,
    transform,
    device,
    perturbation_name,
):
    # Load original RGB image
    original = Image.open(
        record["image_path"]
    ).convert("RGB")

    # Apply perturbation in original
    # image coordinates
    perturbed = apply_perturbation(
        original,
        record["bounding_box"],
        perturbation_name,
    )

    # Verify bbox interior unchanged
    changed_inside = (
        count_changed_bbox_pixels(
            original,
            perturbed,
            record["bounding_box"],
        )
    )

    assert changed_inside == 0

    # Apply identical INTR preprocessing
    original_tensor, _ = transform(
        original,
        None,
    )

    perturbed_tensor, _ = transform(
        perturbed,
        None,
    )

    # Original inference
    (
        original_logits,
        original_attention,
        _,
    ) = run_intr(
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

    # Perturbed inference
    (
        perturbed_logits,
        perturbed_attention,
        _,
    ) = run_intr(
        model,
        perturbed_tensor,
        device,
    )

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

    # Fixed-query attention comparison
    # Always use the ORIGINAL predicted
    # class query for both images.
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

    # Attention similarity
    cosine = cosine_similarity(
        original_query_attention,
        perturbed_query_attention,
    )

    pearson = pearson_correlation(
        original_query_attention,
        perturbed_query_attention,
    )

    top20_iou = topk_iou(
        original_query_attention,
        perturbed_query_attention,
        fraction=TOP_ATTENTION_FRACTION,
    )

    # Fixed original-class margin
    original_margin = class_margin(
        original_logits,
        fixed_query_index,
    )

    perturbed_margin = class_margin(
        perturbed_logits,
        fixed_query_index,
    )

    margin_change = (
        perturbed_margin
        - original_margin
    )

    # A/B/C/D category
    category = classify_abcd(
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


if __name__ == "__main__":

    cli_args = parse_args()

    args = get_intr_config()

    checkpoint_path = Path(
        cli_args.checkpoint
    )

    assert checkpoint_path.exists(), (
        f"Checkpoint not found: "
        f"{checkpoint_path}"
    )

    perturbation_name = (
        cli_args.perturbation
    )

    output_path = Path(
        cli_args.output
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "Perturbation:",
        perturbation_name,
    )

    print(
        "Output:",
        output_path,
    )

    print("Loading INTR...")

    model = load_intr_model(
        args,
        checkpoint_path,
    )

    transform = make_intr_test_transform()

    dataset = CUBDataset(
        root=CUB_ROOT,
        split="test",
        transform=None,
    )

    dataset_lookup = (
        build_dataset_lookup(
            dataset
        )
    )

    # Load frozen formal cohort
    formal_rows = load_formal_set(
        FORMAL_SET_PATH
    )

    print()
    print("=== Frozen Formal Set Check ===")

    print(
        "Formal rows:",
        len(formal_rows),
    )

    formal_image_ids = [
        int(row["image_id"])
        for row in formal_rows
    ]

    print(
        "Unique image IDs:",
        len(set(formal_image_ids)),
    )

    assert len(formal_rows) == 400
    assert len(set(formal_image_ids)) == 400

    # Verify every formal image exists
    # in CUB test metadata
    missing_ids = [
        image_id
        for image_id in formal_image_ids
        if image_id not in dataset_lookup
    ]

    print(
        "Missing image IDs:",
        missing_ids,
    )

    assert not missing_ids

    print()
    print(
        f"=== Running Formal "
        f"{perturbation_name} Evaluation ==="
    )

    results = []

    total = len(
        formal_rows
    )

    for index, formal_row in enumerate(
        formal_rows,
        start=1,
    ):
        image_id = int(
            formal_row["image_id"]
        )

        record = dataset_lookup[
            image_id
        ]

        # Important formal-set sanity:
        # CSV class must match dataset class.
        assert int(
            formal_row["class_id"]
        ) == int(
            record["class_id"]
        )

        result = evaluate_sample(
            model=model,
            record=record,
            transform=transform,
            device=args.device,
            perturbation_name=perturbation_name,
        )

        result[
            "checkpoint_name"
        ] = checkpoint_path.name

        result[
            "baseline_pred_class_id"
        ] = int(
            formal_row[
                "predicted_class_id"
            ]
        )

        result[
            "checkpoint_name"
        ] = checkpoint_path.name

        result[
            "original_prediction_changed_from_baseline"
        ] = (
            result[
                "original_pred_class_id"
            ]
            !=
            int(
                formal_row[
                    "predicted_class_id"
                ]
            )
        )

        results.append(
            result
        )

        print(
            f"{index:3d}/{total} | "
            f"Image {image_id:5d} | "
            f"Class "
            f"{int(record['class_id']):3d} | "
            f"Pred "
            f"{result['original_pred_class_id']:3d}"
            f" -> "
            f"{result['perturbed_pred_class_id']:3d} | "
            f"Cos "
            f"{result['attention_cosine']:.3f} | "
            f"IoU "
            f"{result['attention_top20_iou']:.3f} | "
            f"Category "
            f"{result['category']}"
        )

    # Save
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

        writer.writerows(
            results
        )

    print()
    print("Saved formal results to:")
    print(output_path)

    # Final category summary
    counts = Counter(
        row["category"]
        for row in results
    )

    print()
    print("=== Formal Category Summary ===")

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

    changed_from_baseline = sum(
        row[
            "original_prediction_changed_from_baseline"
        ]
        for row in results
    )

    print()
    print(
        "=== Clean / Perturbed Correctness ==="
    )

    print(
        "Clean correct on frozen formal cohort:",
        f"{clean_correct}/{total}",
        f"({clean_correct / total:.1%})",
    )

    print(
        "Perturbed correct:",
        f"{perturbed_correct}/{total}",
        f"({perturbed_correct / total:.1%})",
    )

    print(
        "Clean predictions changed "
        "from original baseline:",
        f"{changed_from_baseline}/{total}",
        f"({changed_from_baseline / total:.1%})",
    )

    print(
        "Baseline-correct predictions retained:",
        f"{clean_correct}/{total}",
        f"({clean_correct / total:.1%})",
    )

    # Final bbox-integrity check
    all_bbox_preserved = all(
        row[
            "changed_bbox_pixel_values"
        ] == 0
        for row in results
    )

    print()
    print(
        "All bbox interiors preserved:",
        all_bbox_preserved,
    )

    assert all_bbox_preserved
