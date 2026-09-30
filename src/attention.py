from pathlib import Path

import torch

from config import get_intr_config
from model import load_intr_model
from dataset import CUBDataset, CUB_ROOT, make_intr_test_transform
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    args = get_intr_config()

    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "intr_checkpoint_cub_detr_r50.pth"
    )

    print("Loading INTR...")
    model = load_intr_model(args, checkpoint_path)

    transform = make_intr_test_transform()

    dataset = CUBDataset(
        root=CUB_ROOT,
        split="test",
        transform=transform,
    )

    # Use the first test image for inspection
    #image, target = dataset[0]
    sample_index = next(
        i
        for i, record in enumerate(dataset.records)
        if record["image_id"] == 126
    )

    image, target = dataset[sample_index]

    # [C, H, W] -> [B, C, H, W]
    image = image.unsqueeze(0).to(args.device)

    with torch.inference_mode():
        (
            outputs,
            encoder_output,
            hs,
            attention_scores,
            avg_attention_scores,
        ) = model(image)

    logits = outputs["query_logits"]
    predicted_index = logits.argmax(dim=1).item()

    print()
    print("=== Image ===")
    print("Image ID:", target["image_id"])
    print("True class ID:", target["class_id"])
    print("Predicted index:", predicted_index)
    print("Predicted class ID:", predicted_index + 1)

    print()
    print("=== Model Output Shapes ===")
    print("Logits:", logits.shape)
    print("Encoder output:", encoder_output.shape)
    print("Decoder hidden states (hs):", hs.shape)
    print("Attention scores:", attention_scores.shape)
    print("Average attention scores:", avg_attention_scores.shape)


    # --------------------------------------------------
    # Extract final-layer attention for predicted query
    # --------------------------------------------------

    final_avg_attention = avg_attention_scores[-1, 0]

    print()
    print("=== Final Decoder Layer ===")
    print("All class-query attention:", final_avg_attention.shape)

    predicted_attention = final_avg_attention[predicted_index]

    print("Predicted-query attention:", predicted_attention.shape)


    # Spatial size of encoder feature map
    feature_height = encoder_output.shape[-2]
    feature_width = encoder_output.shape[-1]

    print("Feature map height:", feature_height)
    print("Feature map width:", feature_width)
    print(
        "Number of spatial positions:",
        feature_height * feature_width,
    )


    # Convert flattened attention back to spatial grid
    attention_map = predicted_attention.reshape(
        feature_height,
        feature_width,
    )

    print()
    print("=== Predicted-Class Attention Map ===")
    print("Attention map shape:", attention_map.shape)
    print("Minimum attention:", attention_map.min().item())
    print("Maximum attention:", attention_map.max().item())
    print("Mean attention:", attention_map.mean().item())
    print("Sum of attention:", attention_map.sum().item())


    # --------------------------------------------------
    # Visualize attention
    # --------------------------------------------------

    # Upsample raw attention map to model input resolution
    input_height = image.shape[-2]
    input_width = image.shape[-1]

    attention_upsampled = F.interpolate(
        attention_map.unsqueeze(0).unsqueeze(0),
        size=(input_height, input_width),
        mode="bilinear",
        align_corners=False,
    )

    attention_upsampled = (
        attention_upsampled
        .squeeze(0)
        .squeeze(0)
        .cpu()
    )

    # Load original RGB image
    record = dataset.records[sample_index]

    original_image = Image.open(
        record["image_path"]
    ).convert("RGB")

    # Resize it to exactly the model-input dimensions
    resized_image = original_image.resize(
        (input_width, input_height),
        Image.Resampling.BILINEAR,
    )

    # Plot
    plt.figure(figsize=(12, 8))

    plt.imshow(resized_image)

    plt.imshow(
        attention_upsampled.numpy(),
        alpha=0.5,
        cmap="jet",
    )

    plt.axis("off")

    plt.title(
        f"INTR Attention | "
        f"True ID: {target['class_id']} | "
        f"Predicted ID: {predicted_index + 1}"
    )

    plt.tight_layout()
    plt.show()
