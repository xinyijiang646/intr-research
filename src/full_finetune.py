import random
import argparse
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
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


# Full-training configuration

BATCH_SIZE = 2
NUM_EPOCHS = 5

# Small held-out pilot validation set.
LR_NON_BACKBONE = 1e-6
LR_BACKBONE = 2e-7

WEIGHT_DECAY = 1e-6
CLIP_MAX_NORM = 0.1

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "intr_checkpoint_cub_detr_r50.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "full_finetune"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--lambda-attn",
        type=float,
        required=True,
    )

    return parser.parse_args()


# Model output helpers
def extract_logits_and_attention(
    model_output,
):
    (
        out,
        encoder_output,
        hs,
        attention_scores,
        avg_attention_scores,
    ) = model_output

    logits = out[
        "query_logits"
    ]

    final_attention = (
        avg_attention_scores[-1]
    )

    return (
        logits,
        final_attention,
    )


def get_gt_query_attention(
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


# Optimizer
def build_optimizer(
    model,
):
    non_backbone_params = []
    backbone_params = []

    for name, parameter in (
        model.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        if "backbone" in name:
            backbone_params.append(
                parameter
            )
        else:
            non_backbone_params.append(
                parameter
            )

    print(
        "Non-backbone trainable tensors:",
        len(non_backbone_params),
    )

    print(
        "Backbone trainable tensors:",
        len(backbone_params),
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    non_backbone_params,
                "lr":
                    LR_NON_BACKBONE,
            },
            {
                "params":
                    backbone_params,
                "lr":
                    LR_BACKBONE,
            },
        ],
        weight_decay=
            WEIGHT_DECAY,
    )

    return optimizer


# One training epoch
def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    lambda_attn,
):
    model.train()

    ce_values = []
    attn_values = []
    total_values = []

    original_correct = 0
    perturbed_correct = 0

    prediction_stable = 0
    total_samples = 0

    for batch_index, batch in enumerate(
        loader
    ):

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

        # Forward original
        original_output = model(
            original
        )

        (
            original_logits,
            original_attention,
        ) = extract_logits_and_attention(
            original_output
        )

        # Forward perturbed
        perturbed_output = model(
            perturbed
        )

        (
            perturbed_logits,
            perturbed_attention,
        ) = extract_logits_and_attention(
            perturbed_output
        )

        # Classification loss
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

        ce_loss = (
            0.5
            * (
                ce_original
                +
                ce_perturbed
            )
        )

        # Attention consistency
        original_gt_attention = (
            get_gt_query_attention(
                original_attention,
                labels,
            )
        )

        perturbed_gt_attention = (
            get_gt_query_attention(
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

        attn_loss = (
            1.0
            -
            cosine
        ).mean()

        # Total objective
        total_loss = (
            ce_loss
            +
            lambda_attn
            * attn_loss
        )

        # Backward
        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss.backward()

        grad_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                model.parameters(),
                CLIP_MAX_NORM,
            )
        )

        optimizer.step()

        # Metrics
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

        original_correct += int(
            (
                original_pred
                ==
                labels
            )
            .sum()
            .item()
        )

        perturbed_correct += int(
            (
                perturbed_pred
                ==
                labels
            )
            .sum()
            .item()
        )

        prediction_stable += int(
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

        ce_values.append(
            ce_loss.item()
        )

        attn_values.append(
            attn_loss.item()
        )

        total_values.append(
            total_loss.item()
        )

        if (
            batch_index == 0
            or
            (batch_index + 1) % 10 == 0
        ):
            print(
                f"  Batch "
                f"{batch_index + 1:03d}/"
                f"{len(loader):03d} | "
                f"CE "
                f"{ce_loss.item():.6f} | "
                f"Attn "
                f"{attn_loss.item():.6f} | "
                f"Total "
                f"{total_loss.item():.6f} | "
                f"GradNorm "
                f"{float(grad_norm):.6f}"
            )

    return {
        "ce_mean":
            mean(
                ce_values
            ),

        "attn_mean":
            mean(
                attn_values
            ),

        "total_mean":
            mean(
                total_values
            ),

        "original_accuracy":
            original_correct
            / total_samples,

        "perturbed_accuracy":
            perturbed_correct
            / total_samples,

        "prediction_stability":
            prediction_stable
            / total_samples,
    }


# One lambda run
def run_full_training(
    seed,
    lambda_attn,
    train_dataset,
    device,
):
    print()
    print(
        "=" * 70
    )

    print(
        f"Full training run: "
        f"seed = {seed}, "
        f"lambda_attn = {lambda_attn}"
    )

    print(
        "=" * 70
    )

    # Reset seeds
    random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    # Fresh model from SAME checkpoint
    args = get_intr_config()

    args.device = str(
        device
    )

    model = load_intr_model(
        args,
        CHECKPOINT_PATH,
    )

    optimizer = build_optimizer(
        model
    )

    # Reset RNG again after model
    # construction/loading so paired
    # augmentation sequence is matched
    # across lambda runs.
    random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
        collate_fn=
            paired_collate_fn,
    )

    # Train
    train_metrics = None

    history = []

    for epoch in range(NUM_EPOCHS):

        print()
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_attn=lambda_attn,
        )

        history.append(
            {
                "epoch":
                    epoch + 1,

                "train":
                    train_metrics,
            }
        )

        print()
        print(
            f"--- Epoch {epoch + 1} summary ---"
        )

        for key, value in (
            train_metrics.items()
        ):
            print(
                f"{key}: "
                f"{value:.6f}"
            )

        random.seed(
            seed + epoch + 1
        )

        torch.manual_seed(
            seed + epoch + 1
        )

    # Save checkpoint
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    lambda_name = (
        str(lambda_attn)
        .replace(
            ".",
            "p",
        )
    )

    output_path = (
        OUTPUT_DIR
        / (
            "full_finetune_"
            f"seed_{seed}_"
            f"lambda_{lambda_name}.pth"
        )
    )

    torch.save(
        {
            "model":
                model.state_dict(),

            "seed":
                seed,

            "lambda_attn":
                lambda_attn,

            "num_epochs":
                NUM_EPOCHS,

            "history":
                history,

            "train_metrics":
                train_metrics,

            "lr_non_backbone":
                LR_NON_BACKBONE,

            "lr_backbone":
                LR_BACKBONE,

            "weight_decay":
                WEIGHT_DECAY,

            "clip_max_norm":
                CLIP_MAX_NORM,
        },
        output_path,
    )

    print()
    print(
        "--- Train summary ---"
    )

    for key, value in (
        train_metrics.items()
    ):
        print(
            f"{key}: "
            f"{value:.6f}"
        )

    print()
    print(
        "Saved:",
        output_path,
    )

    return {
        "seed":
            seed,

        "lambda_attn":
            lambda_attn,

        "train":
            train_metrics,

        "history":
            history,

        "checkpoint":
            str(
                output_path
            ),
    }


# Main
if __name__ == "__main__":

    args_cli = parse_args()

    seed = (
        args_cli.seed
    )

    lambda_attn = (
        args_cli.lambda_attn
    )

    print(
        "=== Full Fine-Tuning ==="
    )

    print(
        "Seed:",
        seed,
    )

    print(
        "Lambda:",
        lambda_attn,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    train_dataset = (
        PairedCUBTrainDataset()
    )

    print(
        "Training images:",
        len(
            train_dataset
        ),
    )

    assert (
        len(train_dataset)
        == 5994
    )

    result = run_full_training(
        seed=seed,
        lambda_attn=
            lambda_attn,
        train_dataset=
            train_dataset,
        device=device,
    )

    print()
    print(
        "Saved:",
        result[
            "checkpoint"
        ],
    )

    print()
    print(
        "=== Full Fine-Tuning Complete ==="
    )
