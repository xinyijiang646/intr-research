from pathlib import Path
import csv
import statistics
import matplotlib.pyplot as plt
import numpy as np


def find_repo_root():
    """
    Works whether this script is saved in repo root or in src/.
    """
    here = Path(__file__).resolve()

    # case 1: script saved in repo root
    if (here.parent / "results").exists():
        return here.parent

    # case 2: script saved in src/
    if (here.parent.parent / "results").exists():
        return here.parent.parent

    raise FileNotFoundError(
        "Could not find repo root containing 'results/'. "
        "Please place this script in the repo root or src/."
    )


def load_summary_row(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return row


def collect_metric(summary_dir, method_prefix, metric_name):
    """
    Example:
        method_prefix = 'lambda0'
        metric_name = 'cross_cosine_diff_mean'
    """
    values = []

    for seed in [42, 43, 44]:
        file_path = summary_dir / f"{method_prefix}_seed{seed}_final5394_summary.csv"
        row = load_summary_row(file_path)
        values.append(float(row[metric_name]))

    return values


def mean_sd(values):
    mean_val = statistics.mean(values)
    sd_val = statistics.stdev(values)
    return mean_val, sd_val


def add_value_labels(ax, bars, means):
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )


def main():
    root = find_repo_root()
    summary_dir = root / "results" / "final_homogenization"
    output_dir = root / "results" / "final_eval" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ["lambda0", "v1", "v2"]
    method_labels = ["λ0", "V1", "V2"]

    # ---- collect values ----
    cosine_means = []
    cosine_sds = []
    pearson_means = []
    pearson_sds = []

    for method in methods:
        cosine_vals = collect_metric(
            summary_dir,
            method,
            "cross_cosine_diff_mean",
        )
        pearson_vals = collect_metric(
            summary_dir,
            method,
            "cross_pearson_diff_mean",
        )

        c_mean, c_sd = mean_sd(cosine_vals)
        p_mean, p_sd = mean_sd(pearson_vals)

        cosine_means.append(c_mean)
        cosine_sds.append(c_sd)
        pearson_means.append(p_mean)
        pearson_sds.append(p_sd)

    # print numbers to console
    print("=== Figure 2 values ===")
    for label, c_m, c_sd, p_m, p_sd in zip(
        method_labels, cosine_means, cosine_sds, pearson_means, pearson_sds
    ):
        print(
            f"{label}: "
            f"cosine = {c_m:.6f} ± {c_sd:.6f}, "
            f"pearson = {p_m:.6f} ± {p_sd:.6f}"
        )

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)

    colors = ["#b3b3b3", "#4C78A8", "#F58518"]
    x = np.arange(len(method_labels))

    # Panel A: cosine
    ax = axes[0]
    bars = ax.bar(
        x,
        cosine_means,
        yerr=cosine_sds,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        width=0.68,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=12)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Mean cosine similarity", fontsize=12)
    ax.set_title("A  Different-class cross-image cosine", fontsize=14, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    add_value_labels(ax, bars, cosine_means)

    # Panel B: pearson
    ax = axes[1]
    bars = ax.bar(
        x,
        pearson_means,
        yerr=pearson_sds,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        width=0.68,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=12)
    ax.set_ylim(0, 0.55)
    ax.set_ylabel("Mean Pearson correlation", fontsize=12)
    ax.set_title("B  Different-class cross-image Pearson", fontsize=14, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    add_value_labels(ax, bars, pearson_means)

    fig.suptitle(
        "Cross-image attention homogenization on the frozen final cohort\n"
        "means across seeds 42, 43, 44; error bars = ± SD",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_png = output_dir / "figure2_cross_image_attention_homogenization.png"
    out_pdf = output_dir / "figure2_cross_image_attention_homogenization.pdf"

    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.show()

    print()
    print("Saved:")
    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()