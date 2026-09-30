import pandas as pd
from pathlib import Path

base = Path("results/final_eval/analysis")

for name in ["blur", "mask"]:
    path = base / f"figure3_candidates_{name}.csv"
    df = pd.read_csv(path)

    print("\n" + "=" * 80)
    print(f"{name.upper()}")
    print("=" * 80)
    print("Columns:")
    print(df.columns.tolist())

    keep_cols = [
        "image_id",
        "class_id",
        "class_name",
        "category",
        "attention_cosine",
        "attention_pearson",
        "attention_top20_iou",
        "margin_change",
        "original_pred_class_id",
        "perturbed_pred_class_id",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    print("\nTop rows:")
    print(df[keep_cols].head(15).to_string(index=False))