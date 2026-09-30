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

FORMAL_SET_PATH = (
    RESULTS_DIR / "formal_400_samples.csv"
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
        FORMAL_SET_PATH
    )

    assert len(formal_rows) == 400

    formal_ids = {
        int(row["image_id"])
        for row in formal_rows
    }

    assert len(formal_ids) == 400

    print()
    print(
        "=== Frozen Formal Set Check ==="
    )
    print(
        "Formal rows:",
        len(formal_rows),
    )
    print(
        "Unique IDs:",
        len(formal_ids),
    )

    # Model / dataset
    model_args = get_intr_config()

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

                "class_name":
                    formal_row[
                        "class_name"
                    ],

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
            f"{index:3d}/400 | "
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

    # Cross-image pairwise similarity
    #
    # We compare DIFFERENT images.
    # High values close to same-image ~0.98 would
    # suggest template-like collapse.
    pair_rows = []

    similarities_all = []
    similarities_same_class = []
    similarities_diff_class = []
    pearsons_all = []
    pearsons_same_class = []
    pearsons_diff_class = []

    n = len(
        resized_maps
    )

    for i in range(n):
        for j in range(
            i + 1,
            n,
        ):
            row_i = (
                resized_maps[i]
            )

            row_j = (
                resized_maps[j]
            )

            sim = cosine(
                row_i["map"],
                row_j["map"],
            )

            pearson_sim = pearson(
                row_i["map"],
                row_j["map"],
            )

            same_class = (
                row_i["class_id"]
                ==
                row_j["class_id"]
            )

            similarities_all.append(
                sim
            )

            if same_class:
                similarities_same_class.append(
                    sim
                )
            else:
                similarities_diff_class.append(
                    sim
                )

            pair_rows.append(
                {
                    "image_id_1":
                        row_i[
                            "image_id"
                        ],

                    "image_id_2":
                        row_j[
                            "image_id"
                        ],

                    "class_id_1":
                        row_i[
                            "class_id"
                        ],

                    "class_id_2":
                        row_j[
                            "class_id"
                        ],

                    "same_class":
                        same_class,

                    "attention_cosine":
                        sim,

                    "attention_pearson":
                        pearson_sim,
                }
            )
            pearsons_all.append(
                pearson_sim
            )

            if same_class:
                pearsons_same_class.append(
                    pearson_sim
                )
            else:
                pearsons_diff_class.append(
                    pearson_sim
                )

    with pairwise_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                pair_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            pair_rows
        )

    # Summary
    print()
    print(
        "=== Attention Distribution Summary ==="
    )

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
    print(
        "=== BBox Localization Summary ==="
    )

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
        f"{clean_correct}/400",
        f"({clean_correct / 400:.1%})",
    )

    print()
    print(
        "=== Cross-Image Template Similarity ==="
    )

    all_stats = summarize(
        similarities_all
    )

    same_stats = summarize(
        similarities_same_class
    )

    diff_stats = summarize(
        similarities_diff_class
    )

    print(
        "All different-image pairs:"
    )
    print(
        f"  n={len(similarities_all)} | "
        f"mean={all_stats['mean']:.6f} | "
        f"median={all_stats['median']:.6f} | "
        f"min={all_stats['min']:.6f} | "
        f"max={all_stats['max']:.6f}"
    )

    print(
        "Same-class different images:"
    )
    print(
        f"  n={len(similarities_same_class)} | "
        f"mean={same_stats['mean']:.6f} | "
        f"median={same_stats['median']:.6f}"
    )

    print(
        "Different-class images:"
    )
    print(
        f"  n={len(similarities_diff_class)} | "
        f"mean={diff_stats['mean']:.6f} | "
        f"median={diff_stats['median']:.6f}"
    )

    print()
    print(
        "Saved pairwise similarities:"
    )
    print(pairwise_csv)

    print()
    print(
        "=== Cross-Image Pearson Correlation ==="
    )

    pearson_all_stats = summarize(
        pearsons_all
    )

    pearson_same_stats = summarize(
        pearsons_same_class
    )

    pearson_diff_stats = summarize(
        pearsons_diff_class
    )

    print(
        "All different-image pairs:"
    )
    print(
        f"  n={len(pearsons_all)} | "
        f"mean={pearson_all_stats['mean']:.6f} | "
        f"median={pearson_all_stats['median']:.6f} | "
        f"min={pearson_all_stats['min']:.6f} | "
        f"max={pearson_all_stats['max']:.6f}"
    )

    print(
        "Same-class different images:"
    )
    print(
        f"  n={len(pearsons_same_class)} | "
        f"mean={pearson_same_stats['mean']:.6f} | "
        f"median={pearson_same_stats['median']:.6f}"
    )

    print(
        "Different-class images:"
    )
    print(
        f"  n={len(pearsons_diff_class)} | "
        f"mean={pearson_diff_stats['mean']:.6f} | "
        f"median={pearson_diff_stats['median']:.6f}"
    )
    print()
    print(
        "=== Collapse Diagnostic Complete ==="
    )


if __name__ == "__main__":
    main()