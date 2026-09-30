from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FINAL_EVAL = ROOT / "results" / "final_eval"


def load_method_csv(filename, prefix):
    df = pd.read_csv(FINAL_EVAL / filename)

    keep = [
        "image_id",
        "class_id",
        "category",
        "prediction_unchanged",
        "original_correct",
        "perturbed_correct",
        "original_pred_class_id",
        "perturbed_pred_class_id",
        "attention_cosine",
        "attention_pearson",
        "attention_top20_iou",
        "margin_change",
    ]

    df = df[keep].copy()
    rename_map = {
        col: f"{prefix}_{col}"
        for col in keep
        if col != "image_id"
    }
    return df.rename(columns=rename_map)


def build_table(perturbation):
    if perturbation == "blur":
        official_file = "official_intr_blur_dual_final5394.csv"
        lambda0_file = "lambda0_seed42_blur_final5394.csv"
        v1_file = "v1_seed42_blur_final5394.csv"
        v2_file = "v2_seed42_blur_final5394.csv"
    elif perturbation == "mask":
        official_file = "official_intr_mask_final5394.csv"
        lambda0_file = "lambda0_seed42_mask_final5394.csv"
        v1_file = "v1_seed42_mask_final5394.csv"
        v2_file = "v2_seed42_mask_final5394.csv"
    else:
        raise ValueError("perturbation must be 'blur' or 'mask'")

    official = load_method_csv(official_file, "official")
    lambda0 = load_method_csv(lambda0_file, "lambda0")
    v1 = load_method_csv(v1_file, "v1")
    v2 = load_method_csv(v2_file, "v2")

    merged = (
        official
        .merge(lambda0, on="image_id")
        .merge(v1, on="image_id")
        .merge(v2, on="image_id")
    )

    # All four result files refer to the same image/class.
    assert (
        (merged["official_class_id"] == merged["lambda0_class_id"]).all()
        and (merged["official_class_id"] == merged["v1_class_id"]).all()
        and (merged["official_class_id"] == merged["v2_class_id"]).all()
    ), "Class IDs do not match across methods."

    merged["class_id"] = merged["official_class_id"]

    # all prediction stable
    stable_all = (
        merged["official_prediction_unchanged"]
        & merged["lambda0_prediction_unchanged"]
        & merged["v1_prediction_unchanged"]
        & merged["v2_prediction_unchanged"]
    )

    # Official / λ0: B，V1 / V2: A
    pattern_main = (
        (merged["official_category"] == "B")
        & (merged["lambda0_category"] == "B")
        & (merged["v1_category"] == "A")
        & (merged["v2_category"] == "A")
    )

    candidates = merged[stable_all & pattern_main].copy()

    # score：
    # Official bad（cosine low），V2 excellent（cosine high），V1 has improvements
    candidates["score"] = (
        (1.0 - candidates["official_attention_cosine"]) * 2.0
        + (1.0 - candidates["lambda0_attention_cosine"]) * 2.0
        + candidates["v1_attention_cosine"] * 1.0
        + candidates["v2_attention_cosine"] * 1.5
        + candidates["v2_attention_top20_iou"] * 1.0
    )

    # review
    show_cols = [
        "image_id",
        "class_id",

        "official_category",
        "official_attention_cosine",
        "official_attention_pearson",
        "official_attention_top20_iou",

        "lambda0_category",
        "lambda0_attention_cosine",
        "lambda0_attention_pearson",
        "lambda0_attention_top20_iou",

        "v1_category",
        "v1_attention_cosine",
        "v1_attention_pearson",
        "v1_attention_top20_iou",

        "v2_category",
        "v2_attention_cosine",
        "v2_attention_pearson",
        "v2_attention_top20_iou",

        "score",
    ]

    candidates = candidates.sort_values(
        ["score", "v2_attention_cosine", "v1_attention_cosine"],
        ascending=[False, False, False],
    )

    return candidates[show_cols]


def main():
    for perturbation in ["blur", "mask"]:
        print()
        print("=" * 80)
        print(f"TOP FIGURE-3 CANDIDATES ({perturbation.upper()})")
        print("=" * 80)

        table = build_table(perturbation)

        out_csv = FINAL_EVAL / "analysis" / f"figure3_candidates_{perturbation}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_csv, index=False)

        print(f"Total candidates: {len(table)}")
        print(f"Saved: {out_csv}")
        print()

        if len(table) == 0:
            print("No candidates found.")
        else:
            print(table.head(15).to_string(index=False))


if __name__ == "__main__":
    main()