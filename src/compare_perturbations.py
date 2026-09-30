import csv
from collections import Counter
from pathlib import Path
from statistics import mean, median


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

BLUR_PATH = (
    RESULTS_DIR
    / "background_blur_50_seed42.csv"
)

MASK_PATH = (
    RESULTS_DIR
    / "background_mask_50_seed42.csv"
)


def load_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def to_float(row, key):
    return float(row[key])


def summarize_metric(rows, key):
    values = [
        to_float(row, key)
        for row in rows
    ]

    return {
        "mean": mean(values),
        "median": median(values),
    }


def print_category_summary(name, rows):
    counts = Counter(
        row["category"]
        for row in rows
    )

    total = len(rows)

    stable_prediction = (
        counts["A"] + counts["B"]
    )

    attention_unstable = (
        counts["B"] + counts["D"]
    )

    if stable_prediction > 0:
        b_among_stable = (
            counts["B"]
            / stable_prediction
        )
    else:
        b_among_stable = 0.0

    print()
    print(f"=== {name} Category Summary ===")

    for category in ["A", "B", "C", "D"]:
        count = counts[category]

        print(
            f"{category}: "
            f"{count}/{total} "
            f"({count / total:.1%})"
        )

    print(
        "Prediction stable:",
        f"{stable_prediction}/{total}",
        f"({stable_prediction / total:.1%})",
    )

    print(
        "Attention unstable:",
        f"{attention_unstable}/{total}",
        f"({attention_unstable / total:.1%})",
    )

    print(
        "B among prediction-stable:",
        f"{counts['B']}/{stable_prediction}",
        f"({b_among_stable:.1%})",
    )


def print_metric_summary(name, rows):
    metrics = [
        "attention_cosine",
        "attention_pearson",
        "attention_top20_iou",
        "margin_change",
    ]

    print()
    print(f"=== {name} Metric Summary ===")

    for key in metrics:
        summary = summarize_metric(
            rows,
            key,
        )

        print(
            f"{key:24s} | "
            f"mean {summary['mean']:.4f} | "
            f"median {summary['median']:.4f}"
        )


if __name__ == "__main__":

    blur_rows = load_csv(BLUR_PATH)
    mask_rows = load_csv(MASK_PATH)

    # --------------------------------------------------
    # Convert to dictionaries indexed by image ID
    # --------------------------------------------------

    blur_by_id = {
        int(row["image_id"]): row
        for row in blur_rows
    }

    mask_by_id = {
        int(row["image_id"]): row
        for row in mask_rows
    }

    blur_ids = set(blur_by_id.keys())
    mask_ids = set(mask_by_id.keys())

    common_ids = blur_ids & mask_ids

    print("=== Paired Dataset Check ===")

    print(
        "Blur images:",
        len(blur_ids),
    )

    print(
        "Mask images:",
        len(mask_ids),
    )

    print(
        "Common images:",
        len(common_ids),
    )

    print(
        "Same image set:",
        blur_ids == mask_ids,
    )

    # We expect exactly the same images because
    # both experiments used seed 42.
    if blur_ids != mask_ids:

        only_blur = sorted(
            blur_ids - mask_ids
        )

        only_mask = sorted(
            mask_ids - blur_ids
        )

        print(
            "Only in blur:",
            only_blur,
        )

        print(
            "Only in mask:",
            only_mask,
        )

        raise RuntimeError(
            "Blur and mask do not contain "
            "the same image set."
        )

    # --------------------------------------------------
    # Category summaries
    # --------------------------------------------------

    print_category_summary(
        "Background Blur",
        blur_rows,
    )

    print_category_summary(
        "Background Mask",
        mask_rows,
    )

    # --------------------------------------------------
    # Metric summaries
    # --------------------------------------------------

    print_metric_summary(
        "Background Blur",
        blur_rows,
    )

    print_metric_summary(
        "Background Mask",
        mask_rows,
    )

    # --------------------------------------------------
    # Paired category transitions
    # --------------------------------------------------

    transitions = Counter()

    for image_id in sorted(common_ids):

        blur_category = (
            blur_by_id[image_id]["category"]
        )

        mask_category = (
            mask_by_id[image_id]["category"]
        )

        transitions[
            (blur_category, mask_category)
        ] += 1

    print()
    print("=== Category Transitions: Blur -> Mask ===")

    for source in ["A", "B", "C", "D"]:

        for target in ["A", "B", "C", "D"]:

            count = transitions[
                (source, target)
            ]

            if count > 0:
                print(
                    f"{source} -> {target}: "
                    f"{count}"
                )

    # --------------------------------------------------
    # B overlap
    # --------------------------------------------------

    blur_b_ids = {
        image_id
        for image_id, row
        in blur_by_id.items()
        if row["category"] == "B"
    }

    mask_b_ids = {
        image_id
        for image_id, row
        in mask_by_id.items()
        if row["category"] == "B"
    }

    b_overlap = (
        blur_b_ids
        & mask_b_ids
    )

    b_union = (
        blur_b_ids
        | mask_b_ids
    )

    print()
    print("=== B-Case Overlap ===")

    print(
        "Blur B cases:",
        len(blur_b_ids),
    )

    print(
        "Mask B cases:",
        len(mask_b_ids),
    )

    print(
        "B in both perturbations:",
        len(b_overlap),
    )

    if len(blur_b_ids) > 0:
        print(
            "Fraction of blur-B also mask-B:",
            f"{len(b_overlap) / len(blur_b_ids):.1%}",
        )

    if len(mask_b_ids) > 0:
        print(
            "Fraction of mask-B also blur-B:",
            f"{len(b_overlap) / len(mask_b_ids):.1%}",
        )

    if len(b_union) > 0:
        print(
            "B-set Jaccard:",
            f"{len(b_overlap) / len(b_union):.3f}",
        )

    print(
        "B-overlap image IDs:",
        sorted(b_overlap),
    )

    # --------------------------------------------------
    # Prediction stability overlap
    # --------------------------------------------------

    blur_stable_ids = {
        image_id
        for image_id, row
        in blur_by_id.items()
        if row["category"] in {"A", "B"}
    }

    mask_stable_ids = {
        image_id
        for image_id, row
        in mask_by_id.items()
        if row["category"] in {"A", "B"}
    }

    stable_both = (
        blur_stable_ids
        & mask_stable_ids
    )

    print()
    print("=== Prediction Stability Overlap ===")

    print(
        "Prediction stable under blur:",
        len(blur_stable_ids),
    )

    print(
        "Prediction stable under mask:",
        len(mask_stable_ids),
    )

    print(
        "Prediction stable under both:",
        len(stable_both),
    )

    # --------------------------------------------------
    # Among samples prediction-stable under BOTH
    # perturbations, compare attention instability.
    # --------------------------------------------------

    both_stable_attention_unstable_blur = 0
    both_stable_attention_unstable_mask = 0
    both_stable_both_attention_unstable = 0

    for image_id in stable_both:

        blur_category = (
            blur_by_id[image_id]["category"]
        )

        mask_category = (
            mask_by_id[image_id]["category"]
        )

        blur_unstable = (
            blur_category == "B"
        )

        mask_unstable = (
            mask_category == "B"
        )

        if blur_unstable:
            both_stable_attention_unstable_blur += 1

        if mask_unstable:
            both_stable_attention_unstable_mask += 1

        if blur_unstable and mask_unstable:
            both_stable_both_attention_unstable += 1

    print()
    print(
        "=== Attention Stability Among "
        "Prediction-Stable-Under-Both Samples ==="
    )

    n_stable_both = len(stable_both)

    if n_stable_both > 0:

        print(
            "Total stable under both:",
            n_stable_both,
        )

        print(
            "Attention unstable under blur:",
            f"{both_stable_attention_unstable_blur}/"
            f"{n_stable_both}",
            f"({both_stable_attention_unstable_blur / n_stable_both:.1%})",
        )

        print(
            "Attention unstable under mask:",
            f"{both_stable_attention_unstable_mask}/"
            f"{n_stable_both}",
            f"({both_stable_attention_unstable_mask / n_stable_both:.1%})",
        )

        print(
            "Attention unstable under both:",
            f"{both_stable_both_attention_unstable}/"
            f"{n_stable_both}",
            f"({both_stable_both_attention_unstable / n_stable_both:.1%})",
        )

    # --------------------------------------------------
    # Per-image metric differences
    #
    # Positive delta means mask metric is larger
    # than blur metric.
    # --------------------------------------------------

    paired_metrics = [
        "attention_cosine",
        "attention_pearson",
        "attention_top20_iou",
        "margin_change",
    ]

    print()
    print("=== Paired Metric Differences: Mask - Blur ===")

    for key in paired_metrics:

        differences = []

        for image_id in sorted(common_ids):

            blur_value = to_float(
                blur_by_id[image_id],
                key,
            )

            mask_value = to_float(
                mask_by_id[image_id],
                key,
            )

            differences.append(
                mask_value - blur_value
            )

        print(
            f"{key:24s} | "
            f"mean delta {mean(differences):+.4f} | "
            f"median delta {median(differences):+.4f}"
        )

    # --------------------------------------------------
    # Print images that are B under both perturbations
    # --------------------------------------------------

    print()
    print("=== Persistent B Cases ===")

    if not b_overlap:
        print("None")

    else:
        for image_id in sorted(b_overlap):

            blur_row = blur_by_id[image_id]
            mask_row = mask_by_id[image_id]

            print(
                f"Image {image_id:4d} | "
                f"Class {int(blur_row['class_id']):3d} | "
                f"Blur "
                f"Cos {float(blur_row['attention_cosine']):.3f}, "
                f"IoU {float(blur_row['attention_top20_iou']):.3f} | "
                f"Mask "
                f"Cos {float(mask_row['attention_cosine']):.3f}, "
                f"IoU {float(mask_row['attention_top20_iou']):.3f}"
            )