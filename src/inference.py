from pathlib import Path

import torch

from config import get_intr_config
from model import load_intr_model
from dataset import CUBDataset, CUB_ROOT, make_intr_test_transform


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def predict_one(model, image, device="cpu"):
    """
    Run INTR inference on one preprocessed image.

    Returns:
    predicted_index:
        Zero-based index of the predicted class.

    logits:
        Raw logits for all 200 classes.
    """

    # Add a batch dimension:
    # [3, H, W] -> [1, 3, H, W]
    image = image.unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs, _, _, _, _ = model(image)

    logits = outputs["query_logits"]

    predicted_index = logits.argmax(dim=1).item()

    return predicted_index, logits.squeeze(0)


if __name__ == "__main__":
    args = get_intr_config()

    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "intr_checkpoint_cub_detr_r50.pth"
    )

    # Load model
    print("Loading INTR...")
    model = load_intr_model(args, checkpoint_path)

    # Load official test preprocessing
    transform = make_intr_test_transform()

    # Load official CUB test split
    dataset = CUBDataset(
        root=CUB_ROOT,
        split="test",
        transform=transform,
    )

    print()
    print("=== 10-Class Sanity Check ===")

    selected_records = []
    seen_classes = set()

    for i in range(len(dataset)):
        class_id = dataset.records[i]["class_id"]

        if class_id not in seen_classes:
            selected_records.append(i)
            seen_classes.add(class_id)

        if len(selected_records) == 10:
            break


    num_correct = 0

    for i in selected_records:
        image, target = dataset[i]

        predicted_index, logits = predict_one(
            model,
            image,
            device=args.device,
        )

        predicted_class_id = predicted_index + 1
        correct = predicted_class_id == target["class_id"]

        if correct:
            num_correct += 1

        print(
            f"Image {target['image_id']:4d} | "
            f"True ID: {target['class_id']:3d} | "
            f"Pred ID: {predicted_class_id:3d} | "
            f"Correct: {correct}"
        )

    print()
    print(
        f"Accuracy: {num_correct}/10 = "
        f"{num_correct / 10:.1%}"
    )