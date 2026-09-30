import random

import numpy as np
from PIL import Image

from dataset import (
    CUBDataset,
    CUB_ROOT,
)

from perturbations import (
    background_blur,
    count_changed_bbox_pixels,
)

from config import (
    BACKGROUND_BLUR_RADIUS,
)

from paired_transforms import (
    paired_train_transform,
)


NUM_TRIALS = 10
SANITY_SEED = 42


def count_changed_safe_bbox_tensor_values(
    tensor_a,
    tensor_b,
    bbox,
    margin=3,
):
    x, y, width, height = bbox

    x1 = int(
        round(x)
    ) + margin

    y1 = int(
        round(y)
    ) + margin

    x2 = int(
        round(
            x + width
        )
    ) - margin

    y2 = int(
        round(
            y + height
        )
    ) - margin

    _, tensor_height, tensor_width = (
        tensor_a.shape
    )

    x1 = max(
        0,
        min(
            tensor_width,
            x1,
        ),
    )

    x2 = max(
        0,
        min(
            tensor_width,
            x2,
        ),
    )

    y1 = max(
        0,
        min(
            tensor_height,
            y1,
        ),
    )

    y2 = max(
        0,
        min(
            tensor_height,
            y2,
        ),
    )

    if (
        x2 <= x1
        or
        y2 <= y1
    ):
        return None

    region_a = (
        tensor_a[
            :,
            y1:y2,
            x1:x2,
        ]
    )

    region_b = (
        tensor_b[
            :,
            y1:y2,
            x1:x2,
        ]
    )

    changed = (
        region_a
        != region_b
    ).sum().item()

    return changed


if __name__ == "__main__":

    random.seed(
        SANITY_SEED
    )

    dataset = CUBDataset(
        root=CUB_ROOT,
        split="train",
        transform=None,
    )

    # Just use the first train image
    # for this engineering sanity check.
    record = dataset.records[0]

    original = Image.open(
        record["image_path"]
    ).convert("RGB")

    bbox = record[
        "bounding_box"
    ]

    perturbed = background_blur(
        original,
        bbox,
        radius=
            BACKGROUND_BLUR_RADIUS,
    )

    print(
        "=== Paired Train Transform Sanity ==="
    )

    print(
        "Image ID:",
        record["image_id"],
    )

    print(
        "Class ID:",
        record["class_id"],
    )

    print(
        "Original size:",
        original.size,
    )

    print(
        "Bounding box:",
        bbox,
    )

    # Before geometric augmentation:
    # bbox interior must be exact
    changed_inside = (
        count_changed_bbox_pixels(
            original,
            perturbed,
            bbox,
        )
    )

    print()
    print(
        "Changed bbox pixel values "
        "before transform:",
        changed_inside,
    )

    assert changed_inside == 0

    # Multiple independent transform draws
    previous_metadata = None
    different_trial_seen = False

    for trial in range(
        1,
        NUM_TRIALS + 1,
    ):

        (
            original_tensor,
            perturbed_tensor,
            transformed_bbox,
            metadata,
        ) = paired_train_transform(
            original.copy(),
            perturbed.copy(),
            bbox,
        )

        print(
            "Transformed bbox:",
            transformed_bbox,
        )

        safe_changed = (
            count_changed_safe_bbox_tensor_values(
                original_tensor,
                perturbed_tensor,
                transformed_bbox,
                margin=3,
            )
        )

        print(
            "Changed values in transformed "
            "safe bbox interior:",
            safe_changed,
        )

        if safe_changed is not None:
            assert safe_changed == 0
        else:
            print(
                "Safe bbox interior unavailable "
                "(bbox may be too small or cropped out)."
            )

        same_shape = (
            original_tensor.shape
            ==
            perturbed_tensor.shape
        )

        # The two tensors should NOT be
        # identical globally, because the
        # background was perturbed.
        tensors_identical = np.array_equal(
            original_tensor
            .detach()
            .cpu()
            .numpy(),
            perturbed_tensor
            .detach()
            .cpu()
            .numpy(),
        )

        if (
            previous_metadata
            is not None
            and
            metadata
            != previous_metadata
        ):
            different_trial_seen = True

        print()
        print(
            f"--- Trial {trial} ---"
        )

        print(
            "Flip:",
            metadata[
                "horizontal_flip"
            ],
        )

        print(
            "Branch:",
            metadata[
                "branch"
            ],
        )

        print(
            "First resize:",
            metadata[
                "first_resize_scale"
            ],
        )

        print(
            "Crop:",
            metadata[
                "crop"
            ],
        )

        print(
            "Second resize:",
            metadata[
                "second_resize_scale"
            ],
        )

        print(
            "Final PIL size:",
            metadata[
                "final_size"
            ],
        )

        print(
            "Original tensor shape:",
            tuple(
                original_tensor.shape
            ),
        )

        print(
            "Perturbed tensor shape:",
            tuple(
                perturbed_tensor.shape
            ),
        )

        print(
            "Pair tensor shapes match:",
            same_shape,
        )

        print(
            "Entire tensors identical:",
            tensors_identical,
        )

        assert same_shape

        previous_metadata = (
            metadata
        )

    print()
    print(
        "Different augmentation "
        "draw observed across trials:",
        different_trial_seen,
    )

    assert different_trial_seen

    print()
    print(
        "=== Paired Transform Sanity Passed ==="
    )
