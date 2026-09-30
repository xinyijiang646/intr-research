import random
from statistics import mean, median

import torch
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    Subset,
)

from training_dataset import (
    PairedCUBTrainDataset,
    paired_collate_fn,
)

from model import (
    load_intr_model,
    PROJECT_ROOT,
)

from config import (
    get_intr_config,
)


SEED = 42
BATCH_SIZE = 2
NUM_BATCHES = 10


def extract_query_logits_and_attention(
    model_output,
):
    (
        out,
        encoder_output,
        hs,
        attention_scores,
        avg_attention_scores,
    ) = model_output

    query_logits = out[
        "query_logits"
    ]

    final_attention = (
        avg_attention_scores[-1]
    )

    return (
        query_logits,
        final_attention,
    )


def get_gt_attention(
    final_attention,
    labels,
):
    """
    final_attention:
        [B, num_classes, spatial]
    """

    batch_size = (
        final_attention.shape[0]
    )

    batch_indices = torch.arange(
        batch_size,
        device=labels.device,
    )

    return final_attention[
        batch_indices,
        labels,
        :
    ]


def summarize(
    name,
    values,
):
    values = [
        float(x)
        for x in values
    ]

    print(
        f"{name:28s} | "
        f"mean {mean(values):.6f} | "
        f"median {median(values):.6f} | "
        f"min {min(values):.6f} | "
        f"max {max(values):.6f}"
    )


if __name__ == "__main__":

    random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=== Loss Scale Survey ==="
    )

    print(
        "Device:",
        device,
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Number of batches:",
        NUM_BATCHES,
    )

    # Dataset / loader
    dataset = (
        PairedCUBTrainDataset()
    )

    # ----------------------------------
    # Select one image from 20
    # different classes
    # ----------------------------------

    rng = random.Random(
        SEED
    )

    records = (
        dataset
        .base_dataset
        .records
    )

    indices_by_class = {}

    for index, record in enumerate(
        records
    ):
        class_id = int(
            record["class_id"]
        )

        indices_by_class.setdefault(
            class_id,
            [],
        ).append(
            index
        )

    all_class_ids = sorted(
        indices_by_class
    )

    selected_class_ids = (
        rng.sample(
            all_class_ids,
            20,
        )
    )

    selected_indices = []

    for class_id in (
        selected_class_ids
    ):
        selected_indices.append(
            rng.choice(
                indices_by_class[
                    class_id
                ]
            )
        )

    survey_dataset = Subset(
        dataset,
        selected_indices,
    )

    print(
        "Selected class IDs:",
        selected_class_ids,
    )

    loader = DataLoader(
        survey_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=
            paired_collate_fn,
    )

    # Model
    args = get_intr_config()

    args.device = str(
        device
    )

    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "intr_checkpoint_cub_detr_r50.pth"
    )

    model = load_intr_model(
        args,
        checkpoint_path,
    )

    model.eval()

    # Storage
    ce_original_values = []
    ce_perturbed_values = []
    ce_average_values = []

    ce_original_per_sample_values = []
    ce_perturbed_per_sample_values = []
    ce_average_per_sample_values = []

    cosine_values = []
    attention_loss_values = []

    prediction_stable_count = 0
    total_samples = 0

    # Survey
    with torch.no_grad():

        for batch_index, batch in enumerate(
            loader
        ):

            if (
                batch_index
                >= NUM_BATCHES
            ):
                break

            original = (
                batch["original"]
                .to(device)
            )

            perturbed = (
                batch["perturbed"]
                .to(device)
            )

            labels = torch.tensor(
                batch["labels"],
                dtype=torch.long,
                device=device,
            )

            # Original forward
            original_output = model(
                original
            )

            (
                original_logits,
                original_attention,
            ) = extract_query_logits_and_attention(
                original_output
            )

            # Perturbed forward
            perturbed_output = model(
                perturbed
            )

            (
                perturbed_logits,
                perturbed_attention,
            ) = extract_query_logits_and_attention(
                perturbed_output
            )

            # CE
            ce_original = (
                F.cross_entropy(
                    original_logits,
                    labels,
                )
            )

            ce_perturbed = (
                F.cross_entropy(
                    perturbed_logits,
                    labels,
                )
            )

            ce_original_per_sample = (
                F.cross_entropy(
                    original_logits,
                    labels,
                    reduction="none",
                )
            )

            ce_perturbed_per_sample = (
                F.cross_entropy(
                    perturbed_logits,
                    labels,
                    reduction="none",
                )
            )

            ce_average = (
                0.5
                * (
                    ce_original
                    +
                    ce_perturbed
                )
            )

            ce_average_per_sample = (
                0.5
                * (
                    ce_original_per_sample
                    +
                    ce_perturbed_per_sample
                )
            )

            # GT-query attention
            original_gt_attention = (
                get_gt_attention(
                    original_attention,
                    labels,
                )
            )

            perturbed_gt_attention = (
                get_gt_attention(
                    perturbed_attention,
                    labels,
                )
            )

            cosine = (
                F.cosine_similarity(
                    original_gt_attention,
                    perturbed_gt_attention,
                    dim=1,
                    eps=1e-8,
                )
            )

            attention_loss = (
                1.0
                -
                cosine
            ).mean()

            # Prediction stability
            original_pred = (
                original_logits.argmax(
                    dim=1
                )
            )

            perturbed_pred = (
                perturbed_logits.argmax(
                    dim=1
                )
            )

            prediction_stable_count += int(
                (
                    original_pred
                    ==
                    perturbed_pred
                )
                .sum()
                .item()
            )

            total_samples += (
                labels.shape[0]
            )

            # Store
            ce_original_values.append(
                ce_original.item()
            )

            ce_perturbed_values.append(
                ce_perturbed.item()
            )

            ce_average_values.append(
                ce_average.item()
            )

            attention_loss_values.append(
                attention_loss.item()
            )

            cosine_values.extend(
                cosine
                .detach()
                .cpu()
                .tolist()
            )

            ce_original_per_sample_values.extend(
                ce_original_per_sample
                .detach()
                .cpu()
                .tolist()
            )

            ce_perturbed_per_sample_values.extend(
                ce_perturbed_per_sample
                .detach()
                .cpu()
                .tolist()
            )

            ce_average_per_sample_values.extend(
                ce_average_per_sample
                .detach()
                .cpu()
                .tolist()
            )

            print(
                f"Batch "
                f"{batch_index + 1:02d} | "
                f"CE avg "
                f"{ce_average.item():.6f} | "
                f"Attn loss "
                f"{attention_loss.item():.6f} | "
                f"Cos mean "
                f"{cosine.mean().item():.6f}"
            )

    # Summary
    print()
    print(
        "=== Per-Sample CE Summary ==="
    )

    summarize(
        "CE original per sample",
        ce_original_per_sample_values,
    )

    summarize(
        "CE perturbed per sample",
        ce_perturbed_per_sample_values,
    )

    summarize(
        "CE average per sample",
        ce_average_per_sample_values,
    )

    print()
    print(
        "=== Summary ==="
    )

    summarize(
        "CE original",
        ce_original_values,
    )

    summarize(
        "CE perturbed",
        ce_perturbed_values,
    )

    summarize(
        "CE average",
        ce_average_values,
    )

    summarize(
        "Attention cosine",
        cosine_values,
    )

    summarize(
        "Attention loss",
        attention_loss_values,
    )

    prediction_stability = (
        prediction_stable_count
        / total_samples
    )

    print()

    print(
        "Prediction stable:",
        f"{prediction_stable_count}/"
        f"{total_samples}",
        f"({prediction_stability:.1%})",
    )

    # Rough lambda scale reference
    ce_mean = mean(
        ce_average_values
    )

    attn_mean = mean(
        attention_loss_values
    )

    print()
    print(
        "=== Lambda Scale Reference ==="
    )

    print(
        "Mean CE:",
        f"{ce_mean:.6f}",
    )

    print(
        "Mean attention loss:",
        f"{attn_mean:.6f}",
    )

    if attn_mean > 0:

        equal_scale_lambda = (
            ce_mean
            / attn_mean
        )

        print(
            "Lambda making mean "
            "attention term ~= mean CE:",
            f"{equal_scale_lambda:.6f}",
        )

        for fraction in [
            0.1,
            0.25,
            0.5,
        ]:

            candidate = (
                fraction
                * equal_scale_lambda
            )

            print(
                f"Lambda for attention term "
                f"~{fraction:.0%} of mean CE:",
                f"{candidate:.6f}",
            )

    print()
    print(
        "=== Loss Scale Survey Complete ==="
    )
