import csv
from collections import defaultdict
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

FORMAL_SET_PATH = (
    RESULTS_DIR
    / "formal_400_samples.csv"
)

BLUR_PATH = (
    RESULTS_DIR
    / "formal_400_background_blur.csv"
)

MASK_PATH = (
    RESULTS_DIR
    / "formal_400_background_mask.csv"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "formal_400_bbox_attention.csv"
)


def load_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def build_by_id(rows):
    return {
        int(row["image_id"]): row
        for row in rows
    }


def build_dataset_lookup(dataset):
    return {
        int(record["image_id"]): record
        for record in dataset.records
    }


def bbox_to_resized_coords(
    bbox,
    original_width,
    original_height,
    resized_width,
    resized_height,
):
    """
    Convert original-image CUB bbox
    (x, y, width, height)
    into resized image coordinates.
    """

    x, y, width, height = bbox

    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height

    scale_x = (
        resized_width
        / original_width
    )

    scale_y = (
        resized_height
        / original_height
    )

    return (
        x1 * scale_x,
        y1 * scale_y,
        x2 * scale_x,
        y2 * scale_y,
    )


def bbox_to_feature_mask(
    bbox,
    original_size,
    resized_size,
    feature_height,
    feature_width,
):
    """
    Map bbox from original image coordinates
    to encoder feature-grid cells.

    A feature cell is counted as inside bbox
    if its cell center lies inside the bbox.
    """

    original_width, original_height = (
        original_size
    )

    resized_height, resized_width = (
        resized_size
    )

    (
        x1,
        y1,
        x2,
        y2,
    ) = bbox_to_resized_coords(
        bbox=bbox,
        original_width=original_width,
        original_height=original_height,
        resized_width=resized_width,
        resized_height=resized_height,
    )

    # Feature-cell center coordinates
    xs = (
        torch.arange(
            feature_width,
            dtype=torch.float32,
        )
        + 0.5
    ) * (
        resized_width
        / feature_width
    )

    ys = (
        torch.arange(
            feature_height,
            dtype=torch.float32,
        )
        + 0.5
    ) * (
        resized_height
        / feature_height
    )

    grid_y, grid_x = torch.meshgrid(
        ys,
        xs,
        indexing="ij",
    )

    mask = (
        (grid_x >= x1)
        &
        (grid_x <= x2)
        &
        (grid_y >= y1)
        &
        (grid_y <= y2)
    )

    return mask


def compute_bbox_attention_ratio(
    attention_vector,
    encoder_output,
    bbox,
    original_size,
    resized_size,
):
    """
    attention_vector:
        flattened final-layer fixed-query
        attention, shape [H*W]

    encoder_output:
        shape [1, C, H, W]
    """

    feature_height = (
        encoder_output.shape[-2]
    )

    feature_width = (
        encoder_output.shape[-1]
    )

    attention_map = (
        attention_vector
        .reshape(
            feature_height,
            feature_width,
        )
    )

    bbox_mask = (
        bbox_to_feature_mask(
            bbox=bbox,
            original_size=original_size,
            resized_size=resized_size,
            feature_height=feature_height,
            feature_width=feature_width,
        )
    )

    total_mass = (
        attention_map.sum().item()
    )

    inside_mass = (
        attention_map[
            bbox_mask
        ]
        .sum()
        .item()
    )

    outside_mass = (
        attention_map[
            ~bbox_mask
        ]
        .sum()
        .item()
    )

    ratio = (
        inside_mass
        / total_mass
    )

    return (
        ratio,
        inside_mass,
        outside_mass,
        int(bbox_mask.sum().item()),
        feature_height
        * feature_width,
    )


def summarize_group(name, rows):
    if not rows:
        print(
            f"{name}: no samples"
        )
        return

    values = [
        float(
            row[
                "bbox_attention_ratio"
            ]
        )
        for row in rows
    ]

    print(
        f"{name:24s} | "
        f"n={len(values):3d} | "
        f"mean={mean(values):.4f} | "
        f"median={median(values):.4f}"
    )


if __name__ == "__main__":

    formal_rows = load_csv(
        FORMAL_SET_PATH
    )

    blur_rows = load_csv(
        BLUR_PATH
    )

    mask_rows = load_csv(
        MASK_PATH
    )

    formal_by_id = build_by_id(
        formal_rows
    )

    blur_by_id = build_by_id(
        blur_rows
    )

    mask_by_id = build_by_id(
        mask_rows
    )

    assert set(formal_by_id) == set(
        blur_by_id
    )

    assert set(formal_by_id) == set(
        mask_by_id
    )

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

    results = []

    print()
    print(
        "=== Computing Original BBox Attention Ratios ==="
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
            args.device,
        )

        pred_index = (
            logits.argmax().item()
        )

        pred_class_id = (
            pred_index + 1
        )

        # Formal cohort must still be
        # baseline-correct.
        assert (
            pred_class_id
            ==
            int(
                formal_row[
                    "class_id"
                ]
            )
        )

        fixed_query_attention = (
            attention[
                pred_index
            ]
        )

        resized_size = (
            image_tensor.shape[-2],
            image_tensor.shape[-1],
        )

        (
            bbox_ratio,
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

        blur_category = (
            blur_by_id[
                image_id
            ]["category"]
        )

        mask_category = (
            mask_by_id[
                image_id
            ]["category"]
        )

        persistent_b = (
            blur_category == "B"
            and
            mask_category == "B"
        )

        persistent_a = (
            blur_category == "A"
            and
            mask_category == "A"
        )

        results.append(
            {
                "image_id":
                    image_id,

                "class_id":
                    int(
                        formal_row[
                            "class_id"
                        ]
                    ),

                "class_name":
                    formal_row[
                        "class_name"
                    ],

                "blur_category":
                    blur_category,

                "mask_category":
                    mask_category,

                "persistent_B":
                    persistent_b,

                "persistent_A":
                    persistent_a,

                "bbox_attention_ratio":
                    bbox_ratio,

                "bbox_attention_mass":
                    inside_mass,

                "outside_attention_mass":
                    outside_mass,

                "bbox_feature_cells":
                    bbox_cells,

                "total_feature_cells":
                    total_cells,
            }
        )

        print(
            f"{index:3d}/400 | "
            f"Image {image_id:5d} | "
            f"Class "
            f"{int(formal_row['class_id']):3d} | "
            f"BBox attention "
            f"{bbox_ratio:.3f} | "
            f"Blur {blur_category} | "
            f"Mask {mask_category}"
        )

    # Save
    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    print()
    print(
        "Saved bbox-attention results to:"
    )

    print(
        OUTPUT_PATH
    )

    # Group comparisons
    blur_a = [
        row
        for row in results
        if row[
            "blur_category"
        ] == "A"
    ]

    blur_b = [
        row
        for row in results
        if row[
            "blur_category"
        ] == "B"
    ]

    mask_a = [
        row
        for row in results
        if row[
            "mask_category"
        ] == "A"
    ]

    mask_b = [
        row
        for row in results
        if row[
            "mask_category"
        ] == "B"
    ]

    persistent_a_rows = [
        row
        for row in results
        if row[
            "persistent_A"
        ]
    ]

    persistent_b_rows = [
        row
        for row in results
        if row[
            "persistent_B"
        ]
    ]

    print()
    print(
        "=== Original BBox Attention Summary ==="
    )

    summarize_group(
        "All formal images",
        results,
    )

    print()

    summarize_group(
        "Blur A",
        blur_a,
    )

    summarize_group(
        "Blur B",
        blur_b,
    )

    print()

    summarize_group(
        "Mask A",
        mask_a,
    )

    summarize_group(
        "Mask B",
        mask_b,
    )

    print()

    summarize_group(
        "Persistent A",
        persistent_a_rows,
    )

    summarize_group(
        "Persistent B",
        persistent_b_rows,
    )

    print()
    print(
        "=== Analysis Complete ==="
    )
