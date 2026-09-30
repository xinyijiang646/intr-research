import random

from PIL import Image
import torch
import torchvision.transforms.functional as TF


TRAIN_SCALES = [
    480,
    512,
    544,
    576,
    608,
    640,
    672,
    704,
    736,
    768,
    800,
]

SMALL_RESIZE_SCALES = [
    400,
    500,
    600,
]

MAX_SIZE = 1333

NORMALIZE_MEAN = [
    0.485,
    0.456,
    0.406,
]

NORMALIZE_STD = [
    0.229,
    0.224,
    0.225,
]


def get_resize_size(
    image,
    target_short_side,
    max_size=1333,
):
    width, height = image.size

    min_side = min(
        width,
        height,
    )

    max_side = max(
        width,
        height,
    )

    scale = (
        target_short_side
        / min_side
    )

    if (
        max_side * scale
        > max_size
    ):
        scale = (
            max_size
            / max_side
        )

    new_width = round(
        width * scale
    )

    new_height = round(
        height * scale
    )

    return (
        new_height,
        new_width,
    )


def resize_pair(
    image_a,
    image_b,
    target_short_side,
    max_size=1333,
):
    size = get_resize_size(
        image_a,
        target_short_side,
        max_size,
    )

    image_a = TF.resize(
        image_a,
        size,
    )

    image_b = TF.resize(
        image_b,
        size,
    )

    return (
        image_a,
        image_b,
    )


def horizontal_flip_pair(
    image_a,
    image_b,
):
    return (
        TF.hflip(
            image_a
        ),
        TF.hflip(
            image_b
        ),
    )


def random_crop_pair(
    image_a,
    image_b,
    min_size=384,
    max_size=600,
):
    width, height = (
        image_a.size
    )

    crop_width = random.randint(
        min(
            min_size,
            width,
        ),
        min(
            width,
            max_size,
        ),
    )

    crop_height = random.randint(
        min(
            min_size,
            height,
        ),
        min(
            height,
            max_size,
        ),
    )

    max_left = (
        width - crop_width
    )

    max_top = (
        height - crop_height
    )

    left = random.randint(
        0,
        max_left,
    )

    top = random.randint(
        0,
        max_top,
    )

    image_a = TF.crop(
        image_a,
        top,
        left,
        crop_height,
        crop_width,
    )

    image_b = TF.crop(
        image_b,
        top,
        left,
        crop_height,
        crop_width,
    )

    crop_info = {
        "top": top,
        "left": left,
        "height": crop_height,
        "width": crop_width,
    }

    return (
        image_a,
        image_b,
        crop_info,
    )


def to_tensor_and_normalize(
    image,
):
    tensor = TF.to_tensor(
        image
    )

    tensor = TF.normalize(
        tensor,
        mean=NORMALIZE_MEAN,
        std=NORMALIZE_STD,
    )

    return tensor


def paired_train_transform(
    original,
    perturbed,
    bbox,
):
    """
    Apply one shared draw of the INTR-style
    geometric augmentation to both images,
    while tracking the transformed bbox.
    """

    assert (
        original.size
        == perturbed.size
    )

    bbox_xyxy = (
        bbox_xywh_to_xyxy(
            bbox
        )
    )

    metadata = {}

    # Shared horizontal flip
    do_flip = (
        random.random()
        < 0.5
    )

    metadata[
        "horizontal_flip"
    ] = do_flip

    if do_flip:

        image_width = (
            original.size[0]
        )

        (
            original,
            perturbed,
        ) = horizontal_flip_pair(
            original,
            perturbed,
        )

        bbox_xyxy = (
            flip_bbox_horizontally(
                bbox_xyxy,
                image_width,
            )
        )

    # Shared RandomSelect branch
    use_simple_resize = (
        random.random()
        < 0.5
    )

    if use_simple_resize:

        metadata[
            "branch"
        ] = "resize_only"

        resize_scale = (
            random.choice(
                TRAIN_SCALES
            )
        )

        metadata[
            "first_resize_scale"
        ] = resize_scale

        # Save size BEFORE resize
        old_size = (
            original.size
        )

        # Resize ONCE
        (
            original,
            perturbed,
        ) = resize_pair(
            original,
            perturbed,
            resize_scale,
            MAX_SIZE,
        )

        new_size = (
            original.size
        )

        # Track bbox through resize
        bbox_xyxy = resize_bbox(
            bbox_xyxy,
            old_size,
            new_size,
        )

        metadata[
            "crop"
        ] = None

        metadata[
            "second_resize_scale"
        ] = None

    else:

        metadata[
            "branch"
        ] = "resize_crop_resize"

        # First resize
        first_resize_scale = (
            random.choice(
                SMALL_RESIZE_SCALES
            )
        )

        metadata[
            "first_resize_scale"
        ] = first_resize_scale

        old_size = (
            original.size
        )

        (
            original,
            perturbed,
        ) = resize_pair(
            original,
            perturbed,
            first_resize_scale,
            MAX_SIZE,
        )

        new_size = (
            original.size
        )

        bbox_xyxy = resize_bbox(
            bbox_xyxy,
            old_size,
            new_size,
        )

        # Shared crop
        (
            original,
            perturbed,
            crop_info,
        ) = random_crop_pair(
            original,
            perturbed,
            min_size=384,
            max_size=600,
        )

        metadata[
            "crop"
        ] = crop_info

        bbox_xyxy = crop_bbox(
            bbox_xyxy,
            left=crop_info["left"],
            top=crop_info["top"],
            crop_width=crop_info["width"],
            crop_height=crop_info["height"],
        )

        # Second resize
        second_resize_scale = (
            random.choice(
                TRAIN_SCALES
            )
        )

        metadata[
            "second_resize_scale"
        ] = second_resize_scale

        # IMPORTANT:
        # Save crop size BEFORE final resize
        old_size = (
            original.size
        )

        # Resize ONCE
        (
            original,
            perturbed,
        ) = resize_pair(
            original,
            perturbed,
            second_resize_scale,
            MAX_SIZE,
        )

        new_size = (
            original.size
        )

        # Track bbox through final resize
        bbox_xyxy = resize_bbox(
            bbox_xyxy,
            old_size,
            new_size,
        )

    # Final metadata
    metadata[
        "final_size"
    ] = original.size

    # Tensor + normalize
    original_tensor = (
        to_tensor_and_normalize(
            original
        )
    )

    perturbed_tensor = (
        to_tensor_and_normalize(
            perturbed
        )
    )

    assert (
        original_tensor.shape
        == perturbed_tensor.shape
    )

    transformed_bbox = (
        bbox_xyxy_to_xywh(
            bbox_xyxy
        )
    )

    return (
        original_tensor,
        perturbed_tensor,
        transformed_bbox,
        metadata,
    )


def bbox_xywh_to_xyxy(bbox):
    x, y, width, height = bbox

    return (
        float(x),
        float(y),
        float(x + width),
        float(y + height),
    )


def bbox_xyxy_to_xywh(bbox):
    x1, y1, x2, y2 = bbox

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    )


def resize_bbox(
    bbox_xyxy,
    old_size,
    new_size,
):
    """
    old_size/new_size:
        PIL convention (width, height)
    """

    old_width, old_height = old_size
    new_width, new_height = new_size

    scale_x = (
        new_width / old_width
    )

    scale_y = (
        new_height / old_height
    )

    x1, y1, x2, y2 = (
        bbox_xyxy
    )

    return (
        x1 * scale_x,
        y1 * scale_y,
        x2 * scale_x,
        y2 * scale_y,
    )


def flip_bbox_horizontally(
    bbox_xyxy,
    image_width,
):
    x1, y1, x2, y2 = (
        bbox_xyxy
    )

    return (
        image_width - x2,
        y1,
        image_width - x1,
        y2,
    )


def crop_bbox(
    bbox_xyxy,
    left,
    top,
    crop_width,
    crop_height,
):
    x1, y1, x2, y2 = (
        bbox_xyxy
    )

    x1 = x1 - left
    x2 = x2 - left

    y1 = y1 - top
    y2 = y2 - top

    x1 = max(
        0.0,
        min(
            float(crop_width),
            x1,
        ),
    )

    x2 = max(
        0.0,
        min(
            float(crop_width),
            x2,
        ),
    )

    y1 = max(
        0.0,
        min(
            float(crop_height),
            y1,
        ),
    )

    y2 = max(
        0.0,
        min(
            float(crop_height),
            y2,
        ),
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )
