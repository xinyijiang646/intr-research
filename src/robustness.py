from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import get_intr_config
from model import load_intr_model
from dataset import CUBDataset, CUB_ROOT, make_intr_test_transform
from perturbations import background_blur


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_intr(model, image_tensor, device):
    """
    Run INTR and return logits and final-layer average attention.
    """

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        (
            outputs,
            encoder_output,
            _,
            _,
            avg_attention_scores,
        ) = model(image_tensor)

    logits = outputs["query_logits"]

    # [200, spatial]
    final_attention = avg_attention_scores[-1, 0]

    return logits.squeeze(0), final_attention, encoder_output


def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()

    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0),
    ).item()


def pearson_correlation(a, b):
    """
    Pearson correlation between two attention vectors.
    """

    a = a.flatten().float()
    b = b.flatten().float()

    a_centered = a - a.mean()
    b_centered = b - b.mean()

    numerator = torch.sum(a_centered * b_centered)

    denominator = torch.sqrt(
        torch.sum(a_centered ** 2)
        * torch.sum(b_centered ** 2)
    )

    return (numerator / denominator).item()


def topk_iou(a, b, fraction=0.20):
    """
    IoU between the top-attention spatial locations
    of two attention vectors.
    """

    a = a.flatten()
    b = b.flatten()

    assert a.numel() == b.numel()

    k = max(
        1,
        int(round(a.numel() * fraction))
    )

    top_a = torch.topk(a, k=k).indices
    top_b = torch.topk(b, k=k).indices

    mask_a = torch.zeros(
        a.numel(),
        dtype=torch.bool,
    )

    mask_b = torch.zeros(
        b.numel(),
        dtype=torch.bool,
    )

    mask_a[top_a] = True
    mask_b[top_b] = True

    intersection = (mask_a & mask_b).sum().item()
    union = (mask_a | mask_b).sum().item()

    return intersection / union


def class_margin(logits, class_index):
    """
    Logit margin of a fixed class against its strongest competitor.
    """

    class_logit = logits[class_index]

    other_logits = logits.clone()
    other_logits[class_index] = float("-inf")

    strongest_other = other_logits.max()

    return (class_logit - strongest_other).item()


if __name__ == "__main__":
    args = get_intr_config()

    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "intr_checkpoint_cub_detr_r50.pth"
    )

    print("Loading INTR...")
    model = load_intr_model(args, checkpoint_path)

    # Dataset without transform:
    # we need the original RGB image first.
    dataset = CUBDataset(
        root=CUB_ROOT,
        split="test",
        transform=None,
    )

    sample_index = next(
        i
        for i, record in enumerate(dataset.records)
        if record["image_id"] == 126
    )

    record = dataset.records[sample_index]

    original = Image.open(
        record["image_path"]
    ).convert("RGB")

    blurred = background_blur(
        original,
        record["bounding_box"],
        radius=10,
    )

    # Apply exactly the same INTR preprocessing
    transform = make_intr_test_transform()

    original_tensor, _ = transform(original, None)
    blurred_tensor, _ = transform(blurred, None)

    # Run both images
    original_logits, original_attention, original_encoder = run_intr(
        model,
        original_tensor,
        args.device,
    )

    blurred_logits, blurred_attention, blurred_encoder = run_intr(
        model,
        blurred_tensor,
        args.device,
    )

    # Original prediction determines the FIXED query
    original_predicted_index = original_logits.argmax().item()
    blurred_predicted_index = blurred_logits.argmax().item()

    original_predicted_class_id = original_predicted_index + 1
    blurred_predicted_class_id = blurred_predicted_index + 1

    # Same query for BOTH images
    fixed_query = original_predicted_index

    original_query_attention = original_attention[fixed_query]
    blurred_query_attention = blurred_attention[fixed_query]

    attention_cosine = cosine_similarity(
        original_query_attention,
        blurred_query_attention,
    )

    attention_pearson = pearson_correlation(
        original_query_attention,
        blurred_query_attention,
    )

    attention_top20_iou = topk_iou(
        original_query_attention,
        blurred_query_attention,
        fraction=0.20,
    )

    original_margin = class_margin(
        original_logits,
        fixed_query,
    )

    blurred_margin = class_margin(
        blurred_logits,
        fixed_query,
    )

    margin_change = blurred_margin - original_margin

    print()
    print("=== Original vs Background Blur ===")
    print("Image ID:", record["image_id"])
    print("True class ID:", record["class_id"])

    print()
    print("Original predicted class ID:",
          original_predicted_class_id)

    print("Blurred predicted class ID:",
          blurred_predicted_class_id)

    print(
        "Prediction unchanged:",
        original_predicted_index == blurred_predicted_index,
    )

    print()
    print("Fixed query index:", fixed_query)
    print(
        "Original attention shape:",
        original_query_attention.shape,
    )
    print(
        "Blurred attention shape:",
        blurred_query_attention.shape,
    )

    print()
    print("Attention cosine similarity:", attention_cosine)
    print("Attention Pearson correlation:", attention_pearson)
    print("Attention top-20% IoU:", attention_top20_iou)

    print()
    print("=== Fixed-Query Classification Margin ===")
    print("Original margin:", original_margin)
    print("Blurred margin:", blurred_margin)
    print("Margin change:", margin_change)