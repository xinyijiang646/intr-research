from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset

from dataset import (
    CUBDataset,
    CUB_ROOT,
)

from perturbations import (
    background_blur,
)

from config import (
    BACKGROUND_BLUR_RADIUS,
)

from paired_transforms import (
    paired_train_transform,
)

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INTR_ROOT = (
    PROJECT_ROOT
    / "third_party"
    / "INTR"
)

if str(INTR_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(INTR_ROOT),
    )

from util.misc import (
    nested_tensor_from_tensor_list,
)

class ScaleWeightedSubset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        dataset,
        indices,
        scale_weights,
    ):
        self.dataset = dataset
        self.indices = list(
            indices
        )
        self.scale_weights = (
            scale_weights
        )

    def __len__(self):
        return len(
            self.indices
        )

    def __getitem__(
        self,
        item,
    ):
        base_index = (
            self.indices[
                item
            ]
        )

        sample = (
            self.dataset[
                base_index
            ]
        )

        scale_weight = (
            self.scale_weights[
                base_index
            ]
        )

        return (
            sample,
            scale_weight,
        )


def scale_weighted_collate_fn(
    batch,
):
    samples = [
        item[0]
        for item in batch
    ]

    weights = [
        item[1]
        for item in batch
    ]

    collated = (
        paired_collate_fn(
            samples
        )
    )

    collated[
        "scale_weights"
    ] = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    return collated

class PairedCUBTrainDataset(Dataset):
    """
    CUB training dataset that returns
    an original / background-blurred pair
    with shared geometric augmentation.
    """

    def __init__(
        self,
        root=CUB_ROOT,
        blur_radius=BACKGROUND_BLUR_RADIUS,
    ):
        self.base_dataset = CUBDataset(
            root=root,
            split="train",
            transform=None,
        )

        self.blur_radius = blur_radius

    def __len__(self):
        return len(
            self.base_dataset.records
        )

    def __getitem__(
        self,
        index,
    ):
        record = (
            self.base_dataset.records[
                index
            ]
        )

        image_path = Path(
            record["image_path"]
        )

        original = Image.open(
            image_path
        ).convert("RGB")

        bbox = record[
            "bounding_box"
        ]

        perturbed = background_blur(
            original,
            bbox,
            radius=
                self.blur_radius,
        )

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

        class_id = int(
            record["class_id"]
        )

        # CUB class IDs are 1..200,
        # but PyTorch CE labels must be 0..199.
        label = (
            class_id - 1
        )

        return {
            "original":
                original_tensor,

            "perturbed":
                perturbed_tensor,

            "label":
                label,

            "class_id":
                class_id,

            "image_id":
                int(
                    record["image_id"]
                ),

            "image_path":
                str(
                    image_path
                ),

            "original_bbox":
                bbox,

            "transformed_bbox":
                transformed_bbox,

            "transform_metadata":
                metadata,
        }


def paired_collate_fn(
    batch,
):
    original_list = [
        sample["original"]
        for sample in batch
    ]

    perturbed_list = [
        sample["perturbed"]
        for sample in batch
    ]

    original_nested = (
        nested_tensor_from_tensor_list(
            original_list
        )
    )

    perturbed_nested = (
        nested_tensor_from_tensor_list(
            perturbed_list
        )
    )

    labels = [
        sample["label"]
        for sample in batch
    ]

    image_ids = [
        sample["image_id"]
        for sample in batch
    ]

    class_ids = [
        sample["class_id"]
        for sample in batch
    ]

    transformed_bboxes = [
        sample[
            "transformed_bbox"
        ]
        for sample in batch
    ]

    metadata = [
        sample[
            "transform_metadata"
        ]
        for sample in batch
    ]

    return {
        "original":
            original_nested,

        "perturbed":
            perturbed_nested,

        "labels":
            labels,

        "image_ids":
            image_ids,

        "class_ids":
            class_ids,

        "transformed_bboxes":
            transformed_bboxes,

        "transform_metadata":
            metadata,
    }
