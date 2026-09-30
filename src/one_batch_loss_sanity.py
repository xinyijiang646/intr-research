import math
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

# First sanity value only.
# We are NOT tuning lambda here.
LAMBDA_ATTN = 1.0


def extract_query_logits(
    model_output,
):
    """
    INTR model forward is expected to return:

    out,
    encoder_output,
    hs,
    attention_scores,
    avg_attention_scores
    """

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

    return (
        query_logits,
        avg_attention_scores,
    )


def get_final_gt_query_attention(
    avg_attention_scores,
    labels,
):
    """
    Extract final decoder-layer attention
    for each sample's ground-truth class query.

    Expected conceptual shape:
        final attention:
        [B, num_queries, spatial]

    Some INTR implementations may include
    extra dimensions, so we print shape
    before making assumptions.
    """

    final_attention = (
        avg_attention_scores[-1]
    )

    print(
        "Final averaged attention raw shape:",
        tuple(
            final_attention.shape
        ),
    )

    if final_attention.dim() != 3:
        raise RuntimeError(
            "Expected final averaged attention "
            "to have 3 dimensions "
            "[B, num_queries, spatial], "
            f"but got {tuple(final_attention.shape)}"
        )

    batch_size = (
        final_attention.shape[0]
    )

    assert (
        labels.shape[0]
        ==
        batch_size
    )

    batch_indices = torch.arange(
        batch_size,
        device=labels.device,
    )

    gt_attention = (
        final_attention[
            batch_indices,
            labels,
            :
        ]
    )

    return gt_attention


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
        "=== One-Batch Loss Sanity ==="
    )

    print(
        "Device:",
        device,
    )

    # Dataset / loader
    dataset = (
        PairedCUBTrainDataset()
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=
            paired_collate_fn,
    )

    batch = next(
        iter(loader)
    )

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

    print(
        "Image IDs:",
        batch["image_ids"],
    )

    print(
        "Labels:",
        labels.tolist(),
    )

    # Model
    args = get_intr_config()

    args.device = str(device)

    checkpoint_path= (
        PROJECT_ROOT
        / "checkpoints"
        / "intr_checkpoint_cub_detr_r50.pth"
    )

    model = load_intr_model(
        args,
        checkpoint_path,
    )

    model.train()

    # Forward: original
    original_output = model(
        original
    )

    (
        original_logits,
        original_avg_attention,
    ) = extract_query_logits(
        original_output
    )

    print()
    print(
        "Original logits shape:",
        tuple(
            original_logits.shape
        ),
    )

    # Forward: perturbed
    perturbed_output = model(
        perturbed
    )

    (
        perturbed_logits,
        perturbed_avg_attention,
    ) = extract_query_logits(
        perturbed_output
    )

    print(
        "Perturbed logits shape:",
        tuple(
            perturbed_logits.shape
        ),
    )

    assert (
        original_logits.shape
        ==
        perturbed_logits.shape
    )

    assert (
        original_logits.shape[0]
        ==
        labels.shape[0]
    )

    assert (
        original_logits.shape[1]
        == 200
    )

    # Classification losses
    loss_cls_original = (
        F.cross_entropy(
            original_logits,
            labels,
        )
    )

    loss_cls_perturbed = (
        F.cross_entropy(
            perturbed_logits,
            labels,
        )
    )

    loss_cls = (
        0.5
        * (
            loss_cls_original
            +
            loss_cls_perturbed
        )
    )

    # Attention extraction
    original_gt_attention = (
        get_final_gt_query_attention(
            original_avg_attention,
            labels,
        )
    )

    perturbed_gt_attention = (
        get_final_gt_query_attention(
            perturbed_avg_attention,
            labels,
        )
    )

    print()
    print(
        "Original GT attention shape:",
        tuple(
            original_gt_attention.shape
        ),
    )

    print(
        "Perturbed GT attention shape:",
        tuple(
            perturbed_gt_attention.shape
        ),
    )

    assert (
        original_gt_attention.shape
        ==
        perturbed_gt_attention.shape
    )

    # Attention cosine consistency
    cosine_per_sample = (
        F.cosine_similarity(
            original_gt_attention,
            perturbed_gt_attention,
            dim=1,
            eps=1e-8,
        )
    )

    loss_attn = (
        1.0
        -
        cosine_per_sample
    ).mean()

    total_loss = (
        loss_cls
        +
        LAMBDA_ATTN
        * loss_attn
    )

    # Print loss values
    print()
    print(
        "=== Loss Values ==="
    )

    print(
        "CE original:",
        float(
            loss_cls_original.item()
        ),
    )

    print(
        "CE perturbed:",
        float(
            loss_cls_perturbed.item()
        ),
    )

    print(
        "CE average:",
        float(
            loss_cls.item()
        ),
    )

    print(
        "Attention cosine per sample:",
        [
            float(x)
            for x
            in cosine_per_sample
            .detach()
            .cpu()
        ],
    )

    print(
        "Attention consistency loss:",
        float(
            loss_attn.item()
        ),
    )

    print(
        "Total loss:",
        float(
            total_loss.item()
        ),
    )

    # Finite checks
    assert math.isfinite(
        loss_cls.item()
    )

    assert math.isfinite(
        loss_attn.item()
    )

    assert math.isfinite(
        total_loss.item()
    )

    # Cosine should normally stay
    # within [-1, 1] numerically.
    assert torch.all(
        cosine_per_sample
        <= 1.0001
    )

    assert torch.all(
        cosine_per_sample
        >= -1.0001
    )

    # Backward sanity
    model.zero_grad(
        set_to_none=True
    )

    total_loss.backward()

    grad_count = 0
    nonfinite_grad_count = 0
    grad_norm_sum = 0.0

    for parameter in (
        model.parameters()
    ):
        if parameter.grad is None:
            continue

        grad_count += 1

        grad = (
            parameter.grad
            .detach()
        )

        if not torch.isfinite(
            grad
        ).all():
            nonfinite_grad_count += 1

        grad_norm_sum += float(
            grad.norm().item()
        )

    print()
    print(
        "=== Backward Check ==="
    )

    print(
        "Parameters with gradients:",
        grad_count,
    )

    print(
        "Parameters with non-finite gradients:",
        nonfinite_grad_count,
    )

    print(
        "Sum of parameter grad norms:",
        grad_norm_sum,
    )

    assert grad_count > 0
    assert nonfinite_grad_count == 0
    assert grad_norm_sum > 0.0

    print()
    print(
        "=== One-Batch Loss Sanity Passed ==="
    )
