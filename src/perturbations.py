from PIL import Image, ImageFilter
import numpy as np


def background_blur(
    image,
    bounding_box,
    radius=10,
):
    """
    Blur only the region outside the CUB bounding box.

    The bbox interior is copied directly from the original image
    and therefore remains pixel-identical.

    Parameters:
    image:
        PIL RGB image.

    bounding_box:
        CUB bbox in (x, y, width, height) format.

    radius:
        Gaussian blur radius.

    Returns:
    perturbed:
        PIL RGB image.
    """

    image = image.convert("RGB")

    x, y, width, height = bounding_box

    # Convert CUB bbox to integer pixel boundaries.
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(image.width, int(round(x + width)))
    y2 = min(image.height, int(round(y + height)))

    # Blur the entire image first.
    blurred = image.filter(
        ImageFilter.GaussianBlur(radius=radius)
    )

    # Then restore the ORIGINAL bbox interior.
    original_region = image.crop(
        (x1, y1, x2, y2)
    )

    blurred.paste(
        original_region,
        (x1, y1)
    )

    return blurred


def count_changed_bbox_pixels(
    original,
    perturbed,
    bounding_box,
):
    """
    Count changed pixel values inside the preserved bbox.
    """

    original_array = np.asarray(
        original.convert("RGB")
    )

    perturbed_array = np.asarray(
        perturbed.convert("RGB")
    )

    x, y, width, height = bounding_box

    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(original.width, int(round(x + width)))
    y2 = min(original.height, int(round(y + height)))

    original_bbox = original_array[
        y1:y2,
        x1:x2
    ]

    perturbed_bbox = perturbed_array[
        y1:y2,
        x1:x2
    ]

    changed = np.count_nonzero(
        original_bbox != perturbed_bbox
    )

    return changed


def background_mask(
    image,
    bounding_box,
    fill_color=(128, 128, 128),
):
    """
    Replace only the region outside the CUB bounding box
    with a constant color.

    The bbox interior remains pixel-identical.
    """

    image = image.convert("RGB")

    x, y, width, height = bounding_box

    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(image.width, int(round(x + width)))
    y2 = min(image.height, int(round(y + height)))

    # Start from a constant-color image
    masked = Image.new(
        "RGB",
        image.size,
        fill_color,
    )

    # Restore the original bbox interior
    original_region = image.crop(
        (x1, y1, x2, y2)
    )

    masked.paste(
        original_region,
        (x1, y1),
    )

    return masked


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    from PIL import Image

    from dataset import CUBDataset, CUB_ROOT


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

    perturbed = background_blur(
        original,
        record["bounding_box"],
        radius=10,
    )

    changed_inside = count_changed_bbox_pixels(
        original,
        perturbed,
        record["bounding_box"],
    )
    """
    print("=== Background Blur Check ===")
    print("Image ID:", record["image_id"])
    print("Image size:", original.size)
    print("Bounding box:", record["bounding_box"])
    print(
        "Changed pixel values inside bbox:",
        changed_inside,
    )

    assert changed_inside == 0

    print("BBox interior preserved exactly: True")

    # Show original and perturbed image
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(perturbed)
    axes[1].set_title("Background Blur")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    """


    masked = background_mask(
        original,
        record["bounding_box"],
    )

    changed_inside = count_changed_bbox_pixels(
        original,
        masked,
        record["bounding_box"],
    )

    print("=== Background Mask Check ===")
    print("Image ID:", record["image_id"])
    print("Image size:", original.size)
    print("Bounding box:", record["bounding_box"])
    print(
        "Changed pixel values inside bbox:",
        changed_inside,
    )

    assert changed_inside == 0

    print("BBox interior preserved exactly: True")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(masked)
    axes[1].set_title("Background Mask")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()