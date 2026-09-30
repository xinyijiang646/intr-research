from pathlib import Path
import csv
import re

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "final_eval"
    / "analysis"
    / "final_main_paper_table.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"


def parse_mean_std(value):
    """
    Parse either:
        '37.3'
    or:
        '36.5 ± 0.1'
    """
    numbers = re.findall(r"-?\d+(?:\.\d+)?", value)

    mean = float(numbers[0])
    std = float(numbers[1]) if len(numbers) > 1 else 0.0

    return mean, std


def main():
    rows = []

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    methods = ["Official", "lambda0", "V1", "V2"]
    labels = ["Official INTR", "λ0", "V1", "V2"]
    perturbations = ["Blur", "Mask"]

    means = {}
    stds = {}

    for perturbation in perturbations:
        means[perturbation] = []
        stds[perturbation] = []

        for method in methods:
            row = next(
                r
                for r in rows
                if r["Method"] == method
                and r["Perturbation"] == perturbation
            )

            mean, std = parse_mean_std(row["B / stable (%)"])

            means[perturbation].append(mean)
            stds[perturbation].append(std)

    x = np.arange(len(methods))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    blur_bars = ax.bar(
        x - width / 2,
        means["Blur"],
        width,
        yerr=stds["Blur"],
        capsize=3,
        label="Background blur",
    )

    mask_bars = ax.bar(
        x + width / 2,
        means["Mask"],
        width,
        yerr=stds["Mask"],
        capsize=3,
        label="Background mask",
    )

    ax.set_ylabel(
        "Attention-unstable cases among\nprediction-stable samples (%)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylim(0, 52)

    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.bar_label(
        blur_bars,
        labels=[f"{v:.1f}" for v in means["Blur"]],
        padding=3,
        fontsize=9,
    )

    ax.bar_label(
        mask_bars,
        labels=[f"{v:.1f}" for v in means["Mask"]],
        padding=3,
        fontsize=9,
    )

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    png_path = OUTPUT_DIR / "figure1_explanation_robustness.png"
    pdf_path = OUTPUT_DIR / "figure1_explanation_robustness.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print("Saved:")
    print(png_path)
    print(pdf_path)

    plt.show()


if __name__ == "__main__":
    main()
