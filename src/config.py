from argparse import Namespace

PILOT_NUM_SAMPLES = 50
PILOT_RANDOM_SEED = 42


def get_intr_config():
    """
    Configuration for the official pretrained INTR model on CUB-200-2011.

    Architecture values follow the defaults in the official INTR repository
    at commit 69b8a693d774b4805efe1adc1d0c4c4a28200568.
    """

    return Namespace(
        # Backbone
        backbone="resnet50",
        lr_backbone=1.00e-5,
        dilation=False,
        position_embedding="sine",

        # Transformer
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=2048,
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        num_queries=200,
        pre_norm=False,

        # Dataset
        dataset_name="cub",

        # Device
        device="cpu",
    )

# --------------------------------------------------
# Robustness experiment settings
# --------------------------------------------------

BACKGROUND_BLUR_RADIUS = 10

ATTENTION_COSINE_THRESHOLD = 0.90
ATTENTION_TOP20_IOU_THRESHOLD = 0.50

TOP_ATTENTION_FRACTION = 0.20