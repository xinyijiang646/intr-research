import random
import argparse
from pathlib import Path
from statistics import mean

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


# Micro-pilot configuration
SEED = 42

BATCH_SIZE = 2
NUM_EPOCHS = 5

TRAIN_IMAGES_PER_CLASS = 1
PILOT_VAL_NUM_CLASSES = 40

LR_NON_BACKBONE = 1e-6
LR_BACKBONE = 2e-7

WEIGHT_DECAY = 1e-6
CLIP_MAX_NORM = 0.1

LAMBDA_ATTN = 0.01

BETA_VALUES = [
    0.0005,
    0.0015,
    0.005,
]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "intr_checkpoint_cub_detr_r50.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "micro_teacher_anchor_pilot"
)


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


# Fixed train / pilot-val split
def build_fixed_pilot_split(
    dataset,
    seed=42,
):
    """
    Train:
        one image from every CUB class
        -> 200 images total

    Pilot validation:
        one additional, disjoint image
        from 40 randomly chosen classes.

    This is a development-only pilot split.
    It does NOT use formal400 or untouched5394.
    """

    rng = random.Random(
        seed
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

    class_ids = sorted(
        indices_by_class
    )

    assert len(
        class_ids
    ) == 200

    train_indices = []
    remaining_by_class = {}

    for class_id in class_ids:

        candidates = list(
            indices_by_class[
                class_id
            ]
        )

        rng.shuffle(
            candidates
        )

        train_index = (
            candidates[0]
        )

        train_indices.append(
            train_index
        )

        remaining_by_class[
            class_id
        ] = candidates[1:]

    pilot_val_class_ids = (
        rng.sample(
            class_ids,
            PILOT_VAL_NUM_CLASSES,
        )
    )

    val_indices = []

    for class_id in (
        pilot_val_class_ids
    ):

        remaining = (
            remaining_by_class[
                class_id
            ]
        )

        assert len(
            remaining
        ) > 0

        val_indices.append(
            remaining[0]
        )

    assert (
        len(train_indices)
        == 200
    )

    assert (
        len(val_indices)
        ==
        PILOT_VAL_NUM_CLASSES
    )

    assert (
        set(train_indices)
        .isdisjoint(
            set(val_indices)
        )
    )

    return (
        train_indices,
        val_indices,
        pilot_val_class_ids,
    )


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
    teacher,
    loader,
    optimizer,
    device,
    lambda_attn,
    beta_anchor,
):
    model.train()
    teacher.eval()

    ce_values = []
    attn_values = []
    anchor_values = []
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

        # Frozen teacher, clean original only.
        with torch.no_grad():

            teacher_output = teacher(
                original
            )

            (
                _,
                teacher_attention,
            ) = extract_logits_and_attention(
                teacher_output
            )

            teacher_gt_attention = (
                get_gt_query_attention(
                    teacher_attention,
                    labels,
                )
            )

        # Student: clean original.
        original_output = model(
            original
        )

        (
            original_logits,
            original_attention,
        ) = extract_logits_and_attention(
            original_output
        )

        # Student: paired perturbation.
        perturbed_output = model(
            perturbed
        )

        (
            perturbed_logits,
            perturbed_attention,
        ) = extract_logits_and_attention(
            perturbed_output
        )

        # Classification.
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

        # GT-query attention.
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

        # Student original <-> perturbation consistency.
        consistency_cosine = (
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
            consistency_cosine
        ).mean()

        # Student clean attention <-> frozen teacher clean attention.
        anchor_cosine = (
            F.cosine_similarity(
                original_gt_attention,
                teacher_gt_attention,
                dim=1,
                eps=1e-8,
            )
        )

        anchor_loss = (
            1.0
            -
            anchor_cosine
        ).mean()

        total_loss = (
            ce_loss
            +
            lambda_attn
            * attn_loss
            +
            beta_anchor
            * anchor_loss
        )

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

        anchor_values.append(
            anchor_loss.item()
        )

        total_values.append(
            total_loss.item()
        )

        if (
            batch_index == 0
            or
            (batch_index + 1) % 25 == 0
        ):
            print(
                f"  Batch "
                f"{batch_index + 1:03d}/"
                f"{len(loader):03d} | "
                f"CE "
                f"{ce_loss.item():.6f} | "
                f"Attn "
                f"{attn_loss.item():.6f} | "
                f"Anchor "
                f"{anchor_loss.item():.6f} | "
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

        "anchor_mean":
            mean(
                anchor_values
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


# Pilot validation
def evaluate_pilot(
    model,
    teacher,
    loader,
    device,
):
    """
    Development-only held-out pilot evaluation.

    Uses the same paired training-style transform machinery.
    It is NOT the formal paper evaluation.
    """

    model.eval()
    teacher.eval()

    ce_values = []

    consistency_cosines = []
    anchor_cosines = []

    original_correct = 0
    perturbed_correct = 0

    prediction_stable = 0
    total_samples = 0

    with torch.no_grad():

        for batch in loader:

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

            teacher_output = teacher(
                original
            )

            (
                _,
                teacher_attention,
            ) = extract_logits_and_attention(
                teacher_output
            )

            teacher_gt_attention = (
                get_gt_query_attention(
                    teacher_attention,
                    labels,
                )
            )

            original_output = model(
                original
            )

            (
                original_logits,
                original_attention,
            ) = extract_logits_and_attention(
                original_output
            )

            perturbed_output = model(
                perturbed
            )

            (
                perturbed_logits,
                perturbed_attention,
            ) = extract_logits_and_attention(
                perturbed_output
            )

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

            consistency_cosine = (
                F.cosine_similarity(
                    original_gt_attention,
                    perturbed_gt_attention,
                    dim=1,
                    eps=1e-8,
                )
            )

            anchor_cosine = (
                F.cosine_similarity(
                    original_gt_attention,
                    teacher_gt_attention,
                    dim=1,
                    eps=1e-8,
                )
            )

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

            consistency_cosines.extend(
                consistency_cosine
                .cpu()
                .tolist()
            )

            anchor_cosines.extend(
                anchor_cosine
                .cpu()
                .tolist()
            )

    return {
        "ce_mean":
            mean(
                ce_values
            ),

        "cosine_mean":
            mean(
                consistency_cosines
            ),

        "attn_loss_mean":
            1.0
            -
            mean(
                consistency_cosines
            ),

        "anchor_cosine_mean":
            mean(
                anchor_cosines
            ),

        "anchor_loss_mean":
            1.0
            -
            mean(
                anchor_cosines
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


# Filename helper
def float_name(
    value,
):
    return (
        str(value)
        .replace(
            ".",
            "p",
        )
        .replace(
            "-",
            "m",
        )
    )


# One beta pilot
def run_beta_pilot(
    beta_anchor,
    train_dataset,
    val_dataset,
    device,
):
    print()
    print(
        "=" * 72
    )

    print(
        f"Teacher-anchor micro pilot | "
        f"seed={SEED} | "
        f"lambda_attn={LAMBDA_ATTN} | "
        f"beta_anchor={beta_anchor}"
    )

    print(
        "=" * 72
    )

    # Construct both student and teacher from the same
    # original INTR checkpoint.
    random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    args = get_intr_config()

    args.device = str(
        device
    )

    model = load_intr_model(
        args,
        CHECKPOINT_PATH,
    )

    teacher = load_intr_model(
        args,
        CHECKPOINT_PATH,
    )

    teacher.eval()

    for parameter in (
        teacher.parameters()
    ):
        parameter.requires_grad = False

    assert not any(
        parameter.requires_grad
        for parameter in teacher.parameters()
    )

    optimizer = build_optimizer(
        model
    )

    # Reset RNG after BOTH models are constructed.
    # This keeps the augmentation/shuffle stream matched
    # across beta values.
    random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        SEED
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

    # Validation order is fixed and not shuffled.
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=
            paired_collate_fn,
    )

    history = []

    for epoch in range(
        NUM_EPOCHS
    ):

        print()
        print(
            f"Epoch "
            f"{epoch + 1}/{NUM_EPOCHS}"
        )

        train_metrics = (
            train_one_epoch(
                model=model,
                teacher=teacher,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                lambda_attn=
                    LAMBDA_ATTN,
                beta_anchor=
                    beta_anchor,
            )
        )

        val_metrics = (
            evaluate_pilot(
                model=model,
                teacher=teacher,
                loader=val_loader,
                device=device,
            )
        )

        history.append(
            {
                "epoch":
                    epoch + 1,

                "train":
                    train_metrics,

                "val":
                    val_metrics,
            }
        )

        print()
        print(
            f"--- Epoch "
            f"{epoch + 1} summary ---"
        )

        print(
            "Train:"
        )

        for key, value in (
            train_metrics.items()
        ):
            print(
                f"  {key}: "
                f"{value:.6f}"
            )

        print(
            "Pilot val:"
        )

        for key, value in (
            val_metrics.items()
        ):
            print(
                f"  {key}: "
                f"{value:.6f}"
            )

        # Preserve the same epoch-level RNG protocol.
        random.seed(
            SEED + epoch + 1
        )

        torch.manual_seed(
            SEED + epoch + 1
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    beta_name = float_name(
        beta_anchor
    )

    output_path = (
        OUTPUT_DIR
        / (
            "micro_teacher_anchor_"
            f"seed_{SEED}_"
            f"lambda_{float_name(LAMBDA_ATTN)}_"
            f"beta_{beta_name}.pth"
        )
    )

    final_train = (
        history[-1]["train"]
    )

    final_val = (
        history[-1]["val"]
    )

    torch.save(
        {
            "model":
                model.state_dict(),

            "method":
                "teacher_anchor_micro_pilot",

            "seed":
                SEED,

            "lambda_attn":
                LAMBDA_ATTN,

            "beta_anchor":
                beta_anchor,

            "num_epochs":
                NUM_EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "train_images":
                len(train_dataset),

            "pilot_val_images":
                len(val_dataset),

            "teacher_checkpoint":
                str(
                    CHECKPOINT_PATH
                ),

            "student_initial_checkpoint":
                str(
                    CHECKPOINT_PATH
                ),

            "history":
                history,

            "final_train":
                final_train,

            "final_val":
                final_val,

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
        "Saved:",
        output_path,
    )

    return {
        "beta_anchor":
            beta_anchor,

        "final_train":
            final_train,

        "final_val":
            final_val,

        "checkpoint":
            str(
                output_path
            ),
    }


# Main
if __name__ == "__main__":

    print(
        "=== Teacher-Anchor Micro Pilot ==="
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

    print(
        "Lambda attention:",
        LAMBDA_ATTN,
    )

    print(
        "Beta candidates:",
        BETA_VALUES,
    )

    base_dataset = (
        PairedCUBTrainDataset()
    )

    assert (
        len(base_dataset)
        == 5994
    )

    (
        train_indices,
        val_indices,
        pilot_val_class_ids,
    ) = build_fixed_pilot_split(
        base_dataset,
        seed=SEED,
    )

    train_dataset = Subset(
        base_dataset,
        train_indices,
    )

    val_dataset = Subset(
        base_dataset,
        val_indices,
    )

    print(
        "Pilot train images:",
        len(train_dataset),
    )

    print(
        "Pilot val images:",
        len(val_dataset),
    )

    print(
        "Pilot val class IDs:",
        pilot_val_class_ids,
    )

    results = []

    for beta_anchor in (
        BETA_VALUES
    ):
        result = run_beta_pilot(
            beta_anchor=
                beta_anchor,
            train_dataset=
                train_dataset,
            val_dataset=
                val_dataset,
            device=device,
        )

        results.append(
            result
        )

    print()
    print(
        "=" * 72
    )

    print(
        "=== Final Micro-Pilot Comparison ==="
    )

    print(
        "=" * 72
    )

    for result in results:

        beta_anchor = (
            result[
                "beta_anchor"
            ]
        )

        final_val = (
            result[
                "final_val"
            ]
        )

        print()
        print(
            f"beta = "
            f"{beta_anchor}"
        )

        print(
            f"  val cosine          : "
            f"{final_val['cosine_mean']:.6f}"
        )

        print(
            f"  val attn loss       : "
            f"{final_val['attn_loss_mean']:.6f}"
        )

        print(
            f"  val anchor cosine   : "
            f"{final_val['anchor_cosine_mean']:.6f}"
        )

        print(
            f"  val anchor loss     : "
            f"{final_val['anchor_loss_mean']:.6f}"
        )

        print(
            f"  val original acc    : "
            f"{final_val['original_accuracy']:.6f}"
        )

        print(
            f"  val perturbed acc   : "
            f"{final_val['perturbed_accuracy']:.6f}"
        )

        print(
            f"  val pred stability  : "
            f"{final_val['prediction_stability']:.6f}"
        )

        print(
            "  checkpoint          : "
            f"{result['checkpoint']}"
        )

    print()
    print(
        "=== Teacher-Anchor Micro Pilot Complete ==="
    )
