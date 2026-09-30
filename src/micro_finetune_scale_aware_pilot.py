import random
from pathlib import Path
from statistics import mean, median
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    Subset,
)

from training_dataset import (
    PairedCUBTrainDataset,
    paired_collate_fn,
    ScaleWeightedSubset,
    scale_weighted_collate_fn,
)

from model import (
    load_intr_model,
    PROJECT_ROOT,
)

from config import (
    get_intr_config,
)


# Pilot configuration
SEED = 42

BATCH_SIZE = 2
NUM_EPOCHS = 5

TRAIN_IMAGES_PER_CLASS = 1

# Small held-out pilot validation set.
PILOT_VAL_NUM_CLASSES = 40

LR_NON_BACKBONE = 1e-6
LR_BACKBONE = 2e-7

WEIGHT_DECAY = 1e-6
CLIP_MAX_NORM = 0.1

LAMBDA_VALUES = [
    0.01,
]

SCALE_ALPHA = 1.0
USE_SCALE_AWARE = True

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "intr_checkpoint_cub_detr_r50.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "micro_finetune_pilot"
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


def compute_train_scale_weights(
    dataset,
    alpha=1.0,
):
    """
    Compute scale-aware weights using ONLY
    the full CUB training split.

    Scale is measured in ORIGINAL image
    coordinates:
        bbox area / image area.
    
    tau is the median scale over the full
    official CUB training split.

    Normalization is NOT done here.
    It will be computed later using only the
    actual 200 pilot-training images.
    """

    records = (
        dataset
        .base_dataset
        .records
    )

    scales = []

    for record in records:

        with Image.open(
            record["image_path"]
        ) as image:
            image_width, image_height = (
                image.size
            )

        (
            x,
            y,
            bbox_width,
            bbox_height,
        ) = record[
            "bounding_box"
        ]

        scale = (
            float(bbox_width)
            * float(bbox_height)
            /
            (
                float(image_width)
                * float(image_height)
            )
        )

        scales.append(
            scale
        )

    tau = median(
        scales
    )

    raw_weights = []

    for scale in scales:

        smallness = max(
            0.0,
            1.0
            -
            scale / tau,
        )

        raw_weight = (
            1.0
            +
            alpha
            * smallness
        )

        raw_weights.append(
            raw_weight
        )

    print()
    print(
        "=== Raw scale-aware weighting ==="
    )

    print(
        "Alpha:",
        alpha,
    )

    print(
        "Full train images:",
        len(scales),
    )

    print(
        "Train median tau:",
        f"{tau:.6f}",
    )

    print(
        "Raw weight range:",
        f"{min(raw_weights):.6f}",
        "to",
        f"{max(raw_weights):.6f}",
    )

    print(
        "Raw weight mean "
        "(full train):",
        f"{mean(raw_weights):.6f}",
    )

    return (
        raw_weights,
        tau,
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
    weighted_attn_values = []

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

        attn_loss_per_sample = (
            1.0
            -
            cosine
        )

        # Keep the unweighted value for
        # apples-to-apples monitoring with V1.
        attn_loss_unweighted = (
            attn_loss_per_sample
            .mean()
        )

        if USE_SCALE_AWARE:

            scale_weights = (
                batch[
                    "scale_weights"
                ]
                .to(device)
            )

            assert (
                scale_weights.shape
                ==
                attn_loss_per_sample.shape
            )

            attn_loss = (
                scale_weights
                *
                attn_loss_per_sample
            ).mean()

        else:

            attn_loss = (
                attn_loss_unweighted
            )

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
            attn_loss_unweighted.item()
        )

        weighted_attn_values.append(
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

        "weighted_attn_mean":
            mean(
                weighted_attn_values
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


# Fixed pilot validation
def evaluate_pilot(
    model,
    loader,
    device,
):
    """
    Important:
    This uses the same paired training-style
    transform machinery.

    Therefore this is ONLY a micro-pilot
    held-out check, not the formal paper
    evaluation.
    """

    model.eval()

    ce_values = []
    cosine_values = []

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

            cosine = (
                F.cosine_similarity(
                    original_gt_attention,
                    perturbed_gt_attention,
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

            cosine_values.extend(
                cosine
                .detach()
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
                cosine_values
            ),

        "attn_loss_mean":
            1.0
            -
            mean(
                cosine_values
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
def run_pilot(
    lambda_attn,
    train_subset,
    val_subset,
    device,
    scale_tau,
    scale_normalization,
):
    print()
    print(
        "=" * 70
    )

    print(
        f"Pilot run: "
        f"lambda_attn = "
        f"{lambda_attn}"
    )

    print(
        "=" * 70
    )

    # Reset seeds
    random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
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
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=
            train_generator,
        num_workers=0,
        collate_fn=
            scale_weighted_collate_fn,
    )

    # Validation uses fixed ordering.
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=
            paired_collate_fn,
    )

    # Train
    train_metrics = None

    history = []

    for epoch in range(
        NUM_EPOCHS
    ):

        print()
        print(
            f"Epoch "
            f"{epoch + 1}/"
            f"{NUM_EPOCHS}"
        )

        train_metrics = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                lambda_attn=
                    lambda_attn,
            )
        )

        # Use the same deterministic
        # validation augmentation sequence
        # after every epoch.
        random.seed(
            SEED + 1000
        )

        torch.manual_seed(
            SEED + 1000
        )

        val_metrics = (
            evaluate_pilot(
                model=model,
                loader=val_loader,
                device=device,
            )
        )

        # Restore a deterministic
        # epoch-specific training RNG state.
        random.seed(
            SEED + epoch + 1
        )

        torch.manual_seed(
            SEED + epoch + 1
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
            f"{epoch + 1} validation ---"
        )

        print(
            "Val original acc:",
            f"{val_metrics['original_accuracy']:.1%}"
        )

        print(
            "Val blur acc:",
            f"{val_metrics['perturbed_accuracy']:.1%}"
        )

        print(
            "Val prediction stable:",
            f"{val_metrics['prediction_stability']:.1%}"
        )

        print(
            "Val attention cosine:",
            f"{val_metrics['cosine_mean']:.6f}"
        )

        print(
            "Val attention loss:",
            f"{val_metrics['attn_loss_mean']:.6f}"
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

    alpha_name = (
        str(SCALE_ALPHA)
        .replace(
            ".",
            "p",
        )
    )

    output_path = (
        OUTPUT_DIR
        / (
            "micro_finetune_"
            "scaleaware_"
            f"alpha_{alpha_name}_"
            f"lambda_{lambda_name}.pth"
        )
    )

    torch.save(
        {
            "model":
                model.state_dict(),

            "lambda_attn":
                lambda_attn,

            "train_metrics":
                train_metrics,

            "val_metrics":
                val_metrics,

            "history":
                history,

            "seed":
                SEED,

            "lr_non_backbone":
                LR_NON_BACKBONE,

            "lr_backbone":
                LR_BACKBONE,

            "scale_aware":
                USE_SCALE_AWARE,

            "scale_alpha":
                SCALE_ALPHA,

            "scale_tau":
                scale_tau,

            "scale_normalization":
                scale_normalization,
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
        "--- Pilot validation summary ---"
    )

    for key, value in (
        val_metrics.items()
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
        "lambda_attn":
            lambda_attn,

        "train":
            train_metrics,

        "val":
            val_metrics,

        "history":
            history,

        "checkpoint":
            str(
                output_path
            ),
    }


# Main
if __name__ == "__main__":

    print(
        "=== Micro Fine-Tuning "
        "Scale-Aware Pilot ==="
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

    # Base paired CUB training dataset
    base_dataset = (
        PairedCUBTrainDataset()
    )

    # Same fixed pilot split as V1
    (
        train_indices,
        val_indices,
        pilot_val_class_ids,
    ) = build_fixed_pilot_split(
        base_dataset,
        seed=SEED,
    )

    print()
    print(
        "Training images:",
        len(
            train_indices
        ),
    )

    print(
        "Pilot validation images:",
        len(
            val_indices
        ),
    )

    print(
        "Pilot validation class IDs:",
        pilot_val_class_ids,
    )

    # Compute scale-aware RAW weights
    # using full official CUB train split
    (
        raw_scale_weights,
        scale_tau,
    ) = compute_train_scale_weights(
        base_dataset,
        alpha=SCALE_ALPHA,
    )

    assert (
        len(raw_scale_weights)
        ==
        len(base_dataset)
    )

    # Normalize using ONLY the actual
    # 200 pilot-training images.
    #
    # This guarantees:
    #
    # mean(weight over pilot train) = 1
    #
    # so V1 and V2 have the same average
    # attention regularization strength.
    pilot_train_raw_weights = [
        raw_scale_weights[
            index
        ]
        for index in train_indices
    ]

    scale_normalization = mean(
        pilot_train_raw_weights
    )

    scale_weights = [
        raw_weight
        /
        scale_normalization
        for raw_weight in
        raw_scale_weights
    ]

    pilot_train_normalized_weights = [
        scale_weights[
            index
        ]
        for index in train_indices
    ]

    print()
    print(
        "=== Pilot-train "
        "scale normalization ==="
    )

    print(
        "Scale normalization:",
        f"{scale_normalization:.6f}",
    )

    print(
        "Pilot-train normalized "
        "weight mean:",
        f"{mean(pilot_train_normalized_weights):.6f}",
    )

    print(
        "Pilot-train normalized "
        "weight range:",
        f"{min(pilot_train_normalized_weights):.6f}",
        "to",
        f"{max(pilot_train_normalized_weights):.6f}",
    )

    assert abs(
        mean(
            pilot_train_normalized_weights
        )
        -
        1.0
    ) < 1e-8

    # Training subset:
    # same 200 images as V1,
    # but now each sample carries its
    # precomputed normalized scale weight.
    train_subset = (
        ScaleWeightedSubset(
            dataset=base_dataset,
            indices=train_indices,
            scale_weights=
                scale_weights,
        )
    )

    # Validation stays completely
    # unchanged / unweighted.
    val_subset = Subset(
        base_dataset,
        val_indices,
    )

    # Run
    all_results = []

    for lambda_attn in (
        LAMBDA_VALUES
    ):

        result = run_pilot(
            lambda_attn=
                lambda_attn,

            train_subset=
                train_subset,

            val_subset=
                val_subset,

            device=
                device,

            scale_tau=
                scale_tau,

            scale_normalization=
                scale_normalization,
        )

        all_results.append(
            result
        )

    # Final comparison
    print()
    print(
        "=" * 70
    )

    print(
        "=== Scale-Aware "
        "Pilot Comparison ==="
    )

    for result in (
        all_results
    ):

        lambda_attn = (
            result[
                "lambda_attn"
            ]
        )

        train = (
            result[
                "train"
            ]
        )

        val = (
            result[
                "val"
            ]
        )

        print()

        print(
            "Lambda:",
            lambda_attn,
        )

        print(
            "Alpha:",
            SCALE_ALPHA,
        )

        print(
            "  Train CE:",
            f"{train['ce_mean']:.6f}",
        )

        # Unweighted attention loss.
        # Directly comparable with V1.
        print(
            "  Train attn loss "
            "(unweighted):",
            f"{train['attn_mean']:.6f}",
        )

        # Actual V2 objective term.
        print(
            "  Train attn loss "
            "(weighted):",
            f"{train['weighted_attn_mean']:.6f}",
        )

        print(
            "  Train original acc:",
            f"{train['original_accuracy']:.1%}",
        )

        print(
            "  Train blur acc:",
            f"{train['perturbed_accuracy']:.1%}",
        )

        print(
            "  Train prediction stable:",
            f"{train['prediction_stability']:.1%}",
        )

        print(
            "  Val original acc:",
            f"{val['original_accuracy']:.1%}",
        )

        print(
            "  Val blur acc:",
            f"{val['perturbed_accuracy']:.1%}",
        )

        print(
            "  Val prediction stable:",
            f"{val['prediction_stability']:.1%}",
        )

        print(
            "  Val attention cosine:",
            f"{val['cosine_mean']:.6f}",
        )

        print(
            "  Val attention loss:",
            f"{val['attn_loss_mean']:.6f}",
        )

    print()

    print(
        "=== Micro Fine-Tuning "
        "Scale-Aware Pilot Complete ==="
    )
