from pathlib import Path
import sys
import torch


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTR_ROOT = PROJECT_ROOT / "third_party" / "INTR"

sys.path.insert(0, str(INTR_ROOT))

from models.intr import build as build_intr


def load_intr_model(args, checkpoint_path):
    """
    Build the official INTR model and load a pretrained checkpoint.

    Parameters:
    args:
        Configuration expected by the official INTR implementation.

    checkpoint_path:
        Path to the pretrained INTR checkpoint.

    Returns:
    model:
        INTR model loaded with pretrained weights and set to evaluation mode.
    """

    # 1. Construct the official INTR architecture.
    model, _ = build_intr(args)

    # 2. Load the pretrained checkpoint.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=args.device,
        weights_only=False,
    )

    # 3. Load model parameters strictly.
    model.load_state_dict(checkpoint["model"], strict=True)

    # 4. Move model to selected device.
    model.to(args.device)

    # 5. Evaluation mode disables training behaviour such as dropout.
    model.eval()

    return model

if __name__ == "__main__":
    from config import get_intr_config

    args = get_intr_config()

    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "intr_checkpoint_cub_detr_r50.pth"
    )

    print("Building INTR...")
    model = load_intr_model(args, checkpoint_path)

    print("INTR checkpoint loaded successfully.")
    print("Device:", args.device)
    print("Number of class queries:", model.num_queries)

    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_parameters:,}")
