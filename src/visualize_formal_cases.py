import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import (
    get_intr_config,
    BACKGROUND_BLUR_RADIUS,
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
)

from robustness import run_intr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

CASE_PATH = (
    RESULTS_DIR
    / "qualitative_cases.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "qualitative_visualizations"
)


def load_cases(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def build_dataset_lookup(dataset):
    return {
        int(record["image_id"]): record
        for record in dataset.records
    }


def resize_for_display(
    image,
    target_height,
    target_width,
):
    """
    Resize PIL image to exactly the same spatial
    resolution as the model input tensor.
    """

    return image.resize(
        (
            target_width,
            target_height,
        ),
        Image.Resampling.BILINEAR,
    )


def attention_to_image_size(
    attention_vector,
    encoder_output,
    target_height,
    target_width,
):
    """
    Reshape flattened attention to encoder
    feature-map shape, then upsample to the
    model-input image resolution.
    """

    # encoder_output expected:
    # [1, hidden_dim, H, W]
    feature_height = (
        encoder_output.shape[-2]
    )

    feature_width = (
        encoder_output.shape[-1]
    )

    expected_length = (
        feature_height
        * feature_width
    )

    assert (
        attention_vector.numel()
        == expected_length
    )

    attention_map = (
        attention_vector
        .reshape(
            1,
            1,
            feature_height,
            feature_width,
        )
    )

    attention_map = F.interpolate(
        attention_map,
        size=(
            target_height,
            target_width,
        ),
        mode="bilinear",
        align_corners=False,
    )

    return (
        attention_map
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )


def normalize_for_display(
    array,
    shared_max,
):
    """
    Use a shared scale across original,
    blur and mask for one image.

    This avoids making every attention map
    look equally strong by independently
    normalizing each one.
    """

    if shared_max <= 0:
        return array

    return array / shared_max


def plot_case(
    case_row,
    record,
    model,
    transform,
    device,
):
    image_id = int(
        case_row["image_id"]
    )

    case_type = (
        case_row["case_type"]
    )

    # Original image
    original = Image.open(
        record["image_path"]
    ).convert("RGB")

    bbox = record[
        "bounding_box"
    ]

    # Perturbations
    blurred = background_blur(
        original,
        bbox,
        radius=BACKGROUND_BLUR_RADIUS,
    )

    masked = background_mask(
        original,
        bbox,
        fill_color=(128, 128, 128),
    )

    # INTR preprocessing
    original_tensor, _ = transform(
        original,
        None,
    )

    blur_tensor, _ = transform(
        blurred,
        None,
    )

    mask_tensor, _ = transform(
        masked,
        None,
    )

    # Model inference
    (
        original_logits,
        original_attention,
        original_encoder,
    ) = run_intr(
        model,
        original_tensor,
        device,
    )

    (
        blur_logits,
        blur_attention,
        blur_encoder,
    ) = run_intr(
        model,
        blur_tensor,
        device,
    )

    (
        mask_logits,
        mask_attention,
        mask_encoder,
    ) = run_intr(
        model,
        mask_tensor,
        device,
    )

    # Fixed original predicted query
    original_pred_index = (
        original_logits.argmax().item()
    )

    fixed_query_index = (
        original_pred_index
    )

    original_pred_class = (
        original_pred_index + 1
    )

    blur_pred_class = (
        blur_logits.argmax().item()
        + 1
    )

    mask_pred_class = (
        mask_logits.argmax().item()
        + 1
    )

    # These selected A/B cases should
    # have stable predictions.
    print()
    print(
        f"{case_type} | "
        f"Image {image_id}"
    )

    print(
        "Predictions:",
        f"{original_pred_class} -> "
        f"{blur_pred_class} -> "
        f"{mask_pred_class}"
    )

    print(
        "Fixed query index:",
        fixed_query_index,
    )

    # Extract fixed-query attention
    original_query_attention = (
        original_attention[
            fixed_query_index
        ]
    )

    blur_query_attention = (
        blur_attention[
            fixed_query_index
        ]
    )

    mask_query_attention = (
        mask_attention[
            fixed_query_index
        ]
    )

    # Input spatial dimensions
    target_height = (
        original_tensor.shape[-2]
    )

    target_width = (
        original_tensor.shape[-1]
    )

    assert (
        blur_tensor.shape[-2:]
        == original_tensor.shape[-2:]
    )

    assert (
        mask_tensor.shape[-2:]
        == original_tensor.shape[-2:]
    )

    # Upsample attention maps
    original_map = (
        attention_to_image_size(
            original_query_attention,
            original_encoder,
            target_height,
            target_width,
        )
    )

    blur_map = (
        attention_to_image_size(
            blur_query_attention,
            blur_encoder,
            target_height,
            target_width,
        )
    )

    mask_map = (
        attention_to_image_size(
            mask_query_attention,
            mask_encoder,
            target_height,
            target_width,
        )
    )

    # Shared visualization scale
    shared_max = max(
        float(original_map.max()),
        float(blur_map.max()),
        float(mask_map.max()),
    )

    original_map_display = (
        normalize_for_display(
            original_map,
            shared_max,
        )
    )

    blur_map_display = (
        normalize_for_display(
            blur_map,
            shared_max,
        )
    )

    mask_map_display = (
        normalize_for_display(
            mask_map,
            shared_max,
        )
    )

    # Resize RGB images to model-input
    # dimensions for spatial alignment.
    original_display = (
        resize_for_display(
            original,
            target_height,
            target_width,
        )
    )

    blur_display = (
        resize_for_display(
            blurred,
            target_height,
            target_width,
        )
    )

    mask_display = (
        resize_for_display(
            masked,
            target_height,
            target_width,
        )
    )

    # Plot 2 x 3 panel
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
    )

    # Raw images
    axes[0, 0].imshow(
        original_display
    )

    axes[0, 1].imshow(
        blur_display
    )

    axes[0, 2].imshow(
        mask_display
    )

    axes[0, 0].set_title(
        f"Original\nPred: "
        f"{original_pred_class}"
    )

    axes[0, 1].set_title(
        f"Background Blur\nPred: "
        f"{blur_pred_class}"
    )

    axes[0, 2].set_title(
        f"Background Mask\nPred: "
        f"{mask_pred_class}"
    )

    # Attention overlays
    attention_maps = [
        original_map_display,
        blur_map_display,
        mask_map_display,
    ]

    display_images = [
        original_display,
        blur_display,
        mask_display,
    ]

    titles = [
        "Original Attention",
        (
            "Blur Attention\n"
            f"Cos="
            f"{float(case_row['blur_cosine']):.3f}, "
            f"IoU="
            f"{float(case_row['blur_iou']):.3f}"
        ),
        (
            "Mask Attention\n"
            f"Cos="
            f"{float(case_row['mask_cosine']):.3f}, "
            f"IoU="
            f"{float(case_row['mask_iou']):.3f}"
        ),
    ]

    overlay_handle = None

    for column in range(3):

        axes[1, column].imshow(
            display_images[column]
        )

        overlay_handle = (
            axes[1, column].imshow(
                attention_maps[column],
                alpha=0.55,
                cmap="jet",
                vmin=0.0,
                vmax=1.0,
            )
        )

        axes[1, column].set_title(
            titles[column]
        )

    # Remove axes
    for row in axes:
        for axis in row:
            axis.axis("off")

    # Figure title
    class_name = (
        case_row["class_name"]
    )

    fig.suptitle(
        (
            f"{case_type} | "
            f"Image {image_id} | "
            f"{class_name} | "
            f"Fixed Query Class "
            f"{original_pred_class}"
        ),
        fontsize=14,
    )

    # Shared colorbar
    fig.colorbar(
        overlay_handle,
        ax=axes.ravel().tolist(),
        fraction=0.02,
        pad=0.02,
        label="Relative attention intensity",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{case_type}_image_{image_id}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":

    args = get_intr_config()

    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "intr_checkpoint_cub_detr_r50.pth"
    )

    print(
        "Loading INTR..."
    )

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

    cases = load_cases(
        CASE_PATH
    )

    print()
    print(
        "Qualitative cases:",
        len(cases),
    )

    assert len(cases) == 4

    for case_row in cases:

        image_id = int(
            case_row["image_id"]
        )

        assert (
            image_id
            in dataset_lookup
        )

        record = (
            dataset_lookup[
                image_id
            ]
        )

        assert (
            int(case_row["class_id"])
            ==
            int(record["class_id"])
        )

        plot_case(
            case_row=case_row,
            record=record,
            model=model,
            transform=transform,
            device=args.device,
        )

    print()
    print(
        "=== Visualization Complete ==="
    )

    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )