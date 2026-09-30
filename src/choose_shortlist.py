import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE = ROOT / "results" / "final_eval" / "analysis"


def show_shortlist(name):
    path = BASE / f"figure3_candidates_{name}.csv"
    df = pd.read_csv(path)

    # sanity check again
    expected = (
        (df["official_category"] == "B")
        & (df["lambda0_category"] == "B")
        & (df["v1_category"] == "A")
        & (df["v2_category"] == "A")
    )

    df = df[expected].copy()

    # the original candidate script generated score
    df = df.sort_values("score", ascending=False)

    show_cols = [
        "image_id",
        "class_id",

        "official_attention_cosine",
        "official_attention_top20_iou",

        "lambda0_attention_cosine",
        "lambda0_attention_top20_iou",

        "v1_attention_cosine",
        "v1_attention_top20_iou",

        "v2_attention_cosine",
        "v2_attention_top20_iou",

        "score",
    ]

    print()
    print("=" * 120)
    print(name.upper())
    print("=" * 120)

    print(
        df[show_cols]
        .head(20)
        .to_string(index=False)
    )


for perturbation in ["blur", "mask"]:
    show_shortlist(perturbation)