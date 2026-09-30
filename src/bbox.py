from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset import (
    CUBDataset, CUB_ROOT, make_intr_test_transform,
)


if __name__ == "__main__":
    # No transform needed here:
    # we want the original image coordinate system.
    dataset = CUBDataset(
        root=CUB_ROOT,
        split="test",
        transform=None,
    )

    # Use our known correct example: image ID 126
    sample_index = next(
        i
        for i, record in enumerate(dataset.records)
        if record["image_id"] == 126
    )

    record = dataset.records[sample_index]

    image = Image.open(
        record["image_path"]
    ).convert("RGB")

    x, y, width, height = record["bounding_box"]

    x2 = x + width
    y2 = y + height

    print("=== Bounding Box Check ===")
    print("Image ID:", record["image_id"])
    print("Image size:", image.size)
    print("Bounding box (x, y, w, h):", record["bounding_box"])
    print("Top-left:", (x, y))
    print("Bottom-right:", (x2, y2))

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(image)

    rectangle = patches.Rectangle(
        (x, y),
        width,
        height,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )

    ax.add_patch(rectangle)

    ax.set_title(
        f"CUB Bounding Box | "
        f"Image ID: {record['image_id']}"
    )

    ax.axis("off")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # Map bbox to resized model-input coordinates
    # --------------------------------------------------

    transform = make_intr_test_transform()

    transformed_dataset = CUBDataset(
        root=CUB_ROOT,
        split="test",
        transform=transform,
    )

    resized_tensor, _ = transformed_dataset[sample_index]

    resized_height = resized_tensor.shape[-2]
    resized_width = resized_tensor.shape[-1]

    original_width, original_height = image.size

    scale_x = resized_width / original_width
    scale_y = resized_height / original_height

    resized_x1 = x * scale_x
    resized_y1 = y * scale_y
    resized_x2 = x2 * scale_x
    resized_y2 = y2 * scale_y

    print()
    print("=== Resized Coordinate Mapping ===")
    print(
        "Original size:",
        (original_width, original_height),
    )
    print(
        "Resized size:",
        (resized_width, resized_height),
    )

    print("Scale X:", scale_x)
    print("Scale Y:", scale_y)

    print(
        "Original bbox:",
        (x, y, x2, y2),
    )

    print(
        "Resized bbox:",
        (
            resized_x1,
            resized_y1,
            resized_x2,
            resized_y2,
        ),
    )
