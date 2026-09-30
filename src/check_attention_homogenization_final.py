import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median

import torch
import torch.nn.functional as F
from PIL import Image

from config import get_intr_config
from model import load_intr_model
from dataset import (
    CUBDataset,
    CUB_ROOT,
    make_intr_test_transform,
)
from robustness import run_intr

# Reuse the already-validated bbox mapping implementation.
from analyze_bbox_attention import (
    load_csv,
    build_dataset_lookup,
    compute_bbox_attention_ratio,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

FINAL_SET_PATH = (
    RESULTS_DIR / "final_untouched_5394.csv"
)


# Attention distribution metrics
def normalize_attention(attention_vector):
    """
    Convert raw attention weights into a valid
    probability distribution.
    """
    attention = (
        attention_vector
        .detach()
        .float()
        .flatten()
        .cpu()
    )

    attention = torch.clamp(
        attention,
        min=0.0,
    )

    total = attention.sum()

    if total <= 0:
        raise ValueError(
            "Attention mass is non-positive."
        )

    return attention / total


def normalized_entropy(attention_vector):
    """
    Shannon entropy normalized to [0, 1].

    1.0 = perfectly uniform attention.
    Lower = more concentrated / peaked.
    """
    p = normalize_attention(
        attention_vector
    )

    eps = 1e-12

    entropy = -torch.sum(
        p * torch.log(p + eps)
    ).item()

    max_entropy = math.log(
        p.numel()
    )

    return entropy / max_entropy


def attention_distribution_stats(
    attention_vector,
):
    p = normalize_attention(
        attention_vector
    )

    attn_mean = p.mean().item()
    attn_std = p.std(
        unbiased=False
    ).item()

    max_attention = p.max().item()

    cv = (
        attn_std / attn_mean
        if attn_mean > 0
        else float("nan")
    )

    return {
        "normalized_entropy":
            normalized_entropy(p),

        "max_attention":
            max_attention,

        "attention_std":
            attn_std,

        "attention_cv":
            cv,
    }


# Resize attention map for cross-image comparison
def resize_attention_map(
    attention_vector,
    encoder_output,
    output_size=(25, 25),
):
    """
    Convert flattened H*W attention into a
    fixed spatial grid.

    This is ONLY used for cross-image template
    similarity, not for the original robustness
    metrics.
    """

    feature_height = (
        encoder_output.shape[-2]
    )

    feature_width = (
        encoder_output.shape[-1]
    )

    attention_map = (
        attention_vector
        .detach()
        .float()
        .reshape(
            1,
            1,
            feature_height,
            feature_width,
        )
    )

    resized = F.interpolate(
        attention_map,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )

    resized = (
        resized
        .squeeze(0)
        .squeeze(0)
        .flatten()
        .cpu()
    )

    resized = torch.clamp(
        resized,
        min=0.0,
    )

    total = resized.sum()

    if total > 0:
        resized = resized / total

    return resized


def cosine(a, b):
    return F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0),
    ).item()


def pearson(a, b):
    a = a.flatten().float()
    b = b.flatten().float()

    a = a - a.mean()
    b = b - b.mean()

    denom = torch.sqrt(
        torch.sum(a ** 2)
        * torch.sum(b ** 2)
    )

    if denom <= 1e-12:
        return float("nan")

    return (
        torch.sum(a * b) / denom
    ).item()


# Summary helpers
def summarize(values):
    values = [
        float(v)
        for v in values
    ]

    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def print_metric_summary(
    name,
    results,
    key,
):
    values = [
        row[key]
        for row in results
    ]

    stats = summarize(values)

    print(
        f"{name:24s} | "
        f"mean={stats['mean']:.6f} | "
        f"median={stats['median']:.6f} | "
        f"min={stats['min']:.6f} | "
        f"max={stats['max']:.6f}"
    )


# Main
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--name",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            RESULTS_DIR
            / "attention_collapse"
        ),
    )

    args_cli = parser.parse_args()

    checkpoint_path = (
        args_cli.checkpoint
    )

    assert checkpoint_path.exists(), (
        f"Checkpoint not found: "
        f"{checkpoint_path}"
    )

    output_dir = (
        args_cli.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_csv = (
        output_dir
        / f"{args_cli.name}_collapse.csv"
    )

    pairwise_csv = (
        output_dir
        / f"{args_cli.name}_cross_image_pairs.csv"
    )

    # Load frozen formal400
    formal_rows = load_csv(
        FINAL_SET_PATH
    )

    assert len(formal_rows) == 5394

    formal_ids = {
        int(row["image_id"])
        for row in formal_rows
    }

    assert len(formal_ids) == 5394

    print()
    print(
        "=== Frozen Final Untouched Set Check ==="
    )
    print(
        "Final rows:",
        len(formal_rows),
    )
    print(
        "Unique IDs:",
        len(formal_ids),
    )

    # Model / dataset
    model_args = get_intr_config()

    if torch.cuda.is_available():
        model_args.device = "cuda:0"
    else:
        model_args.device = "cpu"

    print()
    print("Evaluation device:", model_args.device)

    print()
    print("Checkpoint:")
    print(checkpoint_path)

    print()
    print("Loading INTR...")

    model = load_intr_model(
        model_args,
        checkpoint_path,
    )

    transform = (
        make_intr_test_transform()
    )

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

    # Per-image original attention analysis
    results = []

    resized_maps = []

    print()
    print(
        "=== Original Attention Collapse Diagnostic ==="
    )

    for index, formal_row in enumerate(
        formal_rows,
        start=1,
    ):
        image_id = int(
            formal_row["image_id"]
        )

        class_id = int(
            formal_row["class_id"]
        )

        record = dataset_lookup[
            image_id
        ]

        original = Image.open(
            record["image_path"]
        ).convert("RGB")

        original_size = (
            original.size
        )

        image_tensor, _ = transform(
            original,
            None,
        )

        (
            logits,
            attention,
            encoder_output,
        ) = run_intr(
            model,
            image_tensor,
            model_args.device,
        )

        pred_index = (
            logits.argmax().item()
        )

        pred_class_id = (
            pred_index + 1
        )

        # Same definition as robustness evaluation:
        # each checkpoint's OWN original prediction query.
        fixed_query_attention = (
            attention[pred_index]
        )

        # Attention distribution
        dist_stats = (
            attention_distribution_stats(
                fixed_query_attention
            )
        )

        # BBox localization
        resized_size = (
            image_tensor.shape[-2],
            image_tensor.shape[-1],
        )

        (
            bbox_attention_ratio,
            inside_mass,
            outside_mass,
            bbox_cells,
            total_cells,
        ) = compute_bbox_attention_ratio(
            attention_vector=
                fixed_query_attention,

            encoder_output=
                encoder_output,

            bbox=
                record[
                    "bounding_box"
                ],

            original_size=
                original_size,

            resized_size=
                resized_size,
        )

        bbox_area_ratio = (
            bbox_cells
            / total_cells
        )

        bbox_enrichment = (
            bbox_attention_ratio
            / bbox_area_ratio
            if bbox_area_ratio > 0
            else float("nan")
        )

        # Standardized map for cross-image test
        resized_map = (
            resize_attention_map(
                fixed_query_attention,
                encoder_output,
                output_size=(25, 25),
            )
        )

        resized_maps.append(
            {
                "image_id":
                    image_id,

                "class_id":
                    class_id,

                "pred_class_id":
                    pred_class_id,

                "map":
                    resized_map,
            }
        )

        results.append(
            {
                "image_id":
                    image_id,

                "class_id":
                    class_id,

                "pred_class_id":
                    pred_class_id,

                "clean_correct":
                    (
                        pred_class_id
                        == class_id
                    ),

                "normalized_entropy":
                    dist_stats[
                        "normalized_entropy"
                    ],

                "max_attention":
                    dist_stats[
                        "max_attention"
                    ],

                "attention_std":
                    dist_stats[
                        "attention_std"
                    ],

                "attention_cv":
                    dist_stats[
                        "attention_cv"
                    ],

                "bbox_attention_mass":
                    bbox_attention_ratio,

                "bbox_area_ratio":
                    bbox_area_ratio,

                "bbox_enrichment":
                    bbox_enrichment,

                "bbox_feature_cells":
                    bbox_cells,

                "total_feature_cells":
                    total_cells,

                "inside_attention_mass":
                    inside_mass,

                "outside_attention_mass":
                    outside_mass,
            }
        )

        print(
            f"{index:4d}/5394 | "
            f"Image {image_id:5d} | "
            f"GT {class_id:3d} | "
            f"Pred {pred_class_id:3d} | "
            f"Entropy "
            f"{dist_stats['normalized_entropy']:.3f} | "
            f"BBox enrich "
            f"{bbox_enrichment:.3f}"
        )

    # Save per-image CSV
    fieldnames = list(
        results[0].keys()
    )

    with output_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print(
        "Saved per-image diagnostic:"
    )
    print(output_csv)


    # Exact cross-image means
    # Mathematically equivalent to all-pair enumeration.
    maps = torch.stack([
        row["map"]
        for row in resized_maps
    ]).float()

    class_ids_tensor = torch.tensor([
        row["class_id"]
        for row in resized_maps
    ])

    def exact_pair_sum(unit_vectors):
        n_vec = unit_vectors.shape[0]

        summed = unit_vectors.sum(dim=0)

        pair_sum = (
            torch.dot(summed, summed)
            - n_vec
        ) / 2.0

        n_pairs = (
            n_vec * (n_vec - 1) // 2
        )

        return pair_sum.item(), n_pairs

    # ---------------- Cosine ----------------

    cos_vecs = F.normalize(
        maps,
        p=2,
        dim=1,
    )

    cos_total_sum, n_all = (
        exact_pair_sum(cos_vecs)
    )

    cos_same_sum = 0.0
    n_same = 0

    for c in torch.unique(class_ids_tensor):

        group = cos_vecs[
            class_ids_tensor == c
        ]

        if len(group) < 2:
            continue

        s, k = exact_pair_sum(group)

        cos_same_sum += s
        n_same += k

    n_diff = n_all - n_same

    cos_diff_sum = (
        cos_total_sum
        - cos_same_sum
    )

    cosine_all_mean = (
        cos_total_sum / n_all
    )

    cosine_same_mean = (
        cos_same_sum / n_same
    )

    cosine_diff_mean = (
        cos_diff_sum / n_diff
    )

    # ---------------- Pearson ----------------

    centered = (
        maps
        - maps.mean(
            dim=1,
            keepdim=True,
        )
    )

    pear_vecs = F.normalize(
        centered,
        p=2,
        dim=1,
    )

    pear_total_sum, _ = (
        exact_pair_sum(pear_vecs)
    )

    pear_same_sum = 0.0

    for c in torch.unique(class_ids_tensor):

        group = pear_vecs[
            class_ids_tensor == c
        ]

        if len(group) < 2:
            continue

        s, _ = exact_pair_sum(group)
        pear_same_sum += s

    pear_diff_sum = (
        pear_total_sum
        - pear_same_sum
    )

    pearson_all_mean = (
        pear_total_sum / n_all
    )

    pearson_same_mean = (
        pear_same_sum / n_same
    )

    pearson_diff_mean = (
        pear_diff_sum / n_diff
    )

    # Summary
    print()
    print("=== Attention Distribution Summary ===")

    print_metric_summary(
        "Normalized entropy",
        results,
        "normalized_entropy",
    )

    print_metric_summary(
        "Max attention",
        results,
        "max_attention",
    )

    print_metric_summary(
        "Attention std",
        results,
        "attention_std",
    )

    print_metric_summary(
        "Attention CV",
        results,
        "attention_cv",
    )

    print()
    print("=== BBox Localization Summary ===")

    print_metric_summary(
        "BBox attention mass",
        results,
        "bbox_attention_mass",
    )

    print_metric_summary(
        "BBox area ratio",
        results,
        "bbox_area_ratio",
    )

    print_metric_summary(
        "BBox enrichment",
        results,
        "bbox_enrichment",
    )

    clean_correct = sum(
        row["clean_correct"]
        for row in results
    )

    print()
    print(
        "Clean correct:",
        f"{clean_correct}/5394",
        f"({clean_correct / 5394:.1%})",
    )

    print()
    print("=== Exact Cross-Image Cosine ===")
    print(
        f"All: n={n_all} | "
        f"mean={cosine_all_mean:.6f}"
    )
    print(
        f"Same class: n={n_same} | "
        f"mean={cosine_same_mean:.6f}"
    )
    print(
        f"Different class: n={n_diff} | "
        f"mean={cosine_diff_mean:.6f}"
    )

    print()
    print("=== Exact Cross-Image Pearson ===")
    print(
        f"All: n={n_all} | "
        f"mean={pearson_all_mean:.6f}"
    )
    print(
        f"Same class: n={n_same} | "
        f"mean={pearson_same_mean:.6f}"
    )
    print(
        f"Different class: n={n_diff} | "
        f"mean={pearson_diff_mean:.6f}"
    )

    # compact summary CSV
    summary_csv = (
        output_dir
        / f"{args_cli.name}_summary.csv"
    )

    summary_row = {
        "name": args_cli.name,
        "n_images": len(results),
        "clean_correct": clean_correct,

        "normalized_entropy_mean":
            mean([
                r["normalized_entropy"]
                for r in results
            ]),

        "max_attention_mean":
            mean([
                r["max_attention"]
                for r in results
            ]),

        "attention_std_mean":
            mean([
                r["attention_std"]
                for r in results
            ]),

        "attention_cv_mean":
            mean([
                r["attention_cv"]
                for r in results
            ]),

        "bbox_attention_mass_mean":
            mean([
                r["bbox_attention_mass"]
                for r in results
            ]),

        "bbox_area_ratio_mean":
            mean([
                r["bbox_area_ratio"]
                for r in results
            ]),

        "bbox_enrichment_mean":
            mean([
                r["bbox_enrichment"]
                for r in results
            ]),

        "cross_cosine_all_mean":
            cosine_all_mean,

        "cross_cosine_same_mean":
            cosine_same_mean,

        "cross_cosine_diff_mean":
            cosine_diff_mean,

        "cross_pearson_all_mean":
            pearson_all_mean,

        "cross_pearson_same_mean":
            pearson_same_mean,

        "cross_pearson_diff_mean":
            pearson_diff_mean,
    }

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary_row.keys()
            ),
        )
        writer.writeheader()
        writer.writerow(summary_row)

    print()
    print("Saved compact summary:")
    print(summary_csv)

    print()
    print(
        "=== Collapse Diagnostic Complete ==="
    )


if __name__ == "__main__":
    main()