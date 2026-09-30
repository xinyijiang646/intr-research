import random

import torch
from torch.utils.data import DataLoader

from training_dataset import (
    PairedCUBTrainDataset,
    paired_collate_fn,
)


SANITY_SEED = 42
BATCH_SIZE = 4
NUM_BATCHES = 3


if __name__ == "__main__":

    random.seed(
        SANITY_SEED
    )

    torch.manual_seed(
        SANITY_SEED
    )

    dataset = (
        PairedCUBTrainDataset()
    )

    print(
        "=== Paired Dataset Sanity ==="
    )

    print(
        "Dataset size:",
        len(dataset),
    )

    assert len(dataset) == 5994

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=
            paired_collate_fn,
    )

    for batch_index, batch in enumerate(
        loader
    ):
        if (
            batch_index
            >= NUM_BATCHES
        ):
            break

        original_nested = (
            batch["original"]
        )

        perturbed_nested = (
            batch["perturbed"]
        )

        original_tensors = (
            original_nested.tensors
        )

        perturbed_tensors = (
            perturbed_nested.tensors
        )

        original_mask = (
            original_nested.mask
        )

        perturbed_mask = (
            perturbed_nested.mask
        )

        print()
        print(
            f"=== Batch {batch_index + 1} ==="
        )

        print(
            "Image IDs:",
            batch["image_ids"],
        )

        print(
            "Class IDs:",
            batch["class_ids"],
        )

        print(
            "Labels:",
            batch["labels"],
        )

        print(
            "Original nested tensor shape:",
            tuple(
                original_tensors.shape
            ),
        )

        print(
            "Perturbed nested tensor shape:",
            tuple(
                perturbed_tensors.shape
            ),
        )

        print(
            "Original mask shape:",
            tuple(
                original_mask.shape
            ),
        )

        print(
            "Perturbed mask shape:",
            tuple(
                perturbed_mask.shape
            ),
        )

        assert (
            original_tensors.shape
            ==
            perturbed_tensors.shape
        )

        assert (
            original_mask.shape
            ==
            perturbed_mask.shape
        )

        assert torch.equal(
            original_mask,
            perturbed_mask,
        )

        # Check label convention
        for (
            class_id,
            label,
        ) in zip(
            batch["class_ids"],
            batch["labels"],
        ):
            assert (
                label
                ==
                class_id - 1
            )

        # Check unique image IDs
        assert (
            len(
                set(
                    batch[
                        "image_ids"
                    ]
                )
            )
            ==
            len(
                batch[
                    "image_ids"
                ]
            )
        )

        print(
            "Pair nested shapes match:",
            True,
        )

        print(
            "Padding masks identical:",
            True,
        )

        print(
            "Labels valid:",
            True,
        )

        print(
            "Unique image IDs:",
            True,
        )

        print(
            "Transformed bboxes:"
        )

        for bbox in (
            batch[
                "transformed_bboxes"
            ]
        ):
            print(
                " ",
                bbox,
            )

    print()
    print(
        "=== Paired Dataset Sanity Passed ==="
    )