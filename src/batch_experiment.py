import csv
import random
from pathlib import Path

from PIL import Image

from config import (
    get_intr_config,
    BACKGROUND_BLUR_RADIUS,
    ATTENTION_COSINE_THRESHOLD,
    ATTENTION_TOP20_IOU_THRESHOLD,
    TOP_ATTENTION_FRACTION,
    PILOT_NUM_SAMPLES,
    PILOT_RANDOM_SEED,
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

PERTURBATION_NAME = "background_mask"


def classify_abcd(
    prediction_unchanged,
    cosine,
    top20_iou,
):
    """
    Assign one of the four robustness categories.

    A: prediction stable   + attention stable
    B: prediction stable   + attention unstable
    C: prediction unstable + attention stable
    D: prediction unstable + attention unstable
    """

    attention_stable = (
        cosine >= ATTENTION_COSINE_THRESHOLD
        and
        top20_iou >= ATTENTION_TOP20_IOU_THRESHOLD
    )

    if prediction_unchanged and attention_stable:
        return "A"

    if prediction_unchanged and not attention_stable:
        return "B"

    if not prediction_unchanged and attention_stable:
        return "C"

    return "D"


def find_correct_samples(
    model,
    dataset,
    transform,
    device,
    num_samples=50,
    seed=42,
):
    """
    Select correctly classified test images from distinct classes.

    Sampling protocol:
    1. Group test images by class.
    2. Shuffle class order with a fixed seed.
    3. Shuffle image order within each class.
    4. Select the first correctly classified image found per class.
    5. Never use perturbation results during sample selection.
    """

    rng = random.Random(seed)

    # Group dataset indices by class
    indices_by_class = {}

    for index, record in enumerate(dataset.records):
        class_id = record["class_id"]

        if class_id not in indices_by_class:
            indices_by_class[class_id] = []

        indices_by_class[class_id].append(index)

    # Randomize class order reproducibly
    class_ids = list(indices_by_class.keys())
    rng.shuffle(class_ids)

    selected = []

    print()
    print("=== Selecting Correctly Classified Samples ===")
    print("Random seed:", seed)

    for class_id in class_ids:

        candidate_indices = indices_by_class[class_id].copy()

        # Randomize images within this class
        rng.shuffle(candidate_indices)

        selected_index = None

        for index in candidate_indices:
            record = dataset.records[index]

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

            predicted_index = logits.argmax().item()
            predicted_class_id = predicted_index + 1

            if predicted_class_id == class_id:
                selected_index = index
                break

        if selected_index is not None:
            selected.append(selected_index)

            record = dataset.records[selected_index]

            print(
                f"Selected {len(selected):2d}/{num_samples} | "
                f"Image {record['image_id']:4d} | "
                f"Class {class_id:3d}"
            )

        if len(selected) == num_samples:
            break

    if len(selected) < num_samples:
        raise RuntimeError(
            f"Could only find {len(selected)} valid samples."
        )

    return selected


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
        f"Unknown perturbation: {perturbation_name}"
    )


def evaluate_sample(
    model,
    dataset,
    sample_index,
    transform,
    device,
    perturbation_name,
):
    record = dataset.records[sample_index]

    original = Image.open(
        record["image_path"]
    ).convert("RGB")

    blurred = apply_perturbation(
        original,
        record["bounding_box"],
        perturbation_name,
    )

    # Verify bbox interior was preserved exactly.
    changed_inside = count_changed_bbox_pixels(
        original,
        blurred,
        record["bounding_box"],
    )

    assert changed_inside == 0

    # Same preprocessing for both.
    original_tensor, _ = transform(
        original,
        None,
    )

    blurred_tensor, _ = transform(
        blurred,
        None,
    )

    # Run INTR.
    original_logits, original_attention, _ = run_intr(
        model,
        original_tensor,
        device,
    )

    perturbed_logits, perturbed_attention, _ = run_intr(
        model,
        blurred_tensor,
        device,
    )

    original_pred_index = original_logits.argmax().item()
    perturbed_pred_index = perturbed_logits.argmax().item()

    original_pred_class_id = original_pred_index + 1
    perturbed_pred_class_id = perturbed_pred_index + 1

    prediction_unchanged = (
        original_pred_index == perturbed_pred_index
    )

    # always use the ORIGINAL predicted query.
    fixed_query = original_pred_index

    original_query_attention = (
        original_attention[fixed_query]
    )

    perturbed_query_attention = (
        perturbed_attention[fixed_query]
    )

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

    original_margin = class_margin(
        original_logits,
        fixed_query,
    )

    blurred_margin = class_margin(
        perturbed_logits,
        fixed_query,
    )

    margin_change = (
        blurred_margin - original_margin
    )

    category = classify_abcd(
        prediction_unchanged,
        cosine,
        top20_iou,
    )

    return {
        "image_id": record["image_id"],
        "class_id": record["class_id"],
        "class_name": record["class_name"],

        "perturbation":
            perturbation_name,

        "original_pred_class_id":
            original_pred_class_id,

        "perturbed_pred_class_id":
            perturbed_pred_class_id,

        "prediction_unchanged":
            prediction_unchanged,

        "fixed_query_index":
            fixed_query,

        "original_margin":
            original_margin,

        "perturbed_margin":
            blurred_margin,

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

    # Raw-image dataset.
    dataset = CUBDataset(
        root=CUB_ROOT,
        split="test",
        transform=None,
    )

    selected_indices = find_correct_samples(
        model=model,
        dataset=dataset,
        transform=transform,
        device=args.device,
        num_samples=PILOT_NUM_SAMPLES,
        seed=PILOT_RANDOM_SEED,
    )

    print()
    print(f"=== Running {PERTURBATION_NAME} Pilot ===")

    results = []

    for number, sample_index in enumerate(
        selected_indices,
        start=1,
    ):
        result = evaluate_sample(
            model=model,
            dataset=dataset,
            sample_index=sample_index,
            transform=transform,
            device=args.device,
            perturbation_name=PERTURBATION_NAME,
        )

        results.append(result)

        print(
            f"{number:2d}/{len(selected_indices)} | "
            f"Image {result['image_id']:4d} | "
            f"Class {result['class_id']:3d} | "
            f"Pred "
            f"{result['original_pred_class_id']:3d}"
            f" -> "
            f"{result['perturbed_pred_class_id']:3d} | "
            f"Cos {result['attention_cosine']:.3f} | "
            f"IoU {result['attention_top20_iou']:.3f} | "
            f"Category {result['category']}"
        )

    # Save CSV
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / f"{PERTURBATION_NAME}_{PILOT_NUM_SAMPLES}_seed{PILOT_RANDOM_SEED}.csv"
    )

    with output_path.open(
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
    print("Saved results to:")
    print(output_path)

    # Category summary
    category_counts = {
        "A": 0,
        "B": 0,
        "C": 0,
        "D": 0,
    }

    for result in results:
        category_counts[result["category"]] += 1

    print()
    print("=== Category Summary ===")

    for category in ["A", "B", "C", "D"]:
        print(
            f"{category}: "
            f"{category_counts[category]}/"
            f"{len(results)}"
        )