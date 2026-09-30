# Robustness of Intrinsic Explanations in INTR

This repository contains the code, experiment metadata, and results for a research project investigating the **robustness of intrinsic attention explanations in INTR** on the CUB-200-2011 fine-grained bird classification dataset.

The project is based on the ICLR 2024 paper **"A Simple Interpretable Transformer for Fine-Grained Image Classification and Analysis (INTR)"**. INTR associates class-specific transformer queries with image regions through decoder cross-attention, allowing its attention maps to serve as intrinsic explanations for classification decisions.

Our main research question is:

> **How robust are INTR's intrinsic explanations under background perturbations that preserve the object bounding box?**

In particular, we investigate cases where the model's prediction remains unchanged while its attention explanation changes substantially.

---

## 1. Research Motivation

An intrinsic explanation is directly involved in the model's prediction process, but this does not necessarily guarantee that the explanation is robust.

For an image \(x\) and a background-perturbed version \(\tilde{x}\), we examine both:

- **Prediction robustness:** whether the predicted class remains unchanged.
- **Explanation robustness:** whether the attention map associated with the original predicted class remains similar.

This leads to four possible outcomes:

| Category | Prediction | Attention |
|---|---|---|
| A | Stable | Stable |
| B | Stable | Unstable |
| C | Unstable | Stable |
| D | Unstable | Unstable |

Category **B** is particularly important because the classification decision is preserved while the intrinsic explanation changes.

---

## 2. Dataset and Baseline

Experiments use **CUB-200-2011**, containing 11,788 bird images from 200 classes.

The official split contains:

- 5,994 training images
- 5,794 test images

We first reproduced the pretrained INTR model on the complete test set.

| Metric | Reproduced result |
|---|---:|
| Top-1 Accuracy | 71.78% |
| Top-5 Accuracy | 89.42% |

The test set was subsequently divided into:

- `formal_400_samples.csv`: 400 images used during method development and diagnostics
- `final_untouched_5394.csv`: 5,394 images reserved for frozen final evaluation

The final 5,394-image cohort was not used to select the proposed methods or their hyperparameters.

---

## 3. Background Perturbations

Two bounding-box-preserving perturbations are evaluated.

### Background Blur

The complete image is blurred using a Gaussian blur, after which the original pixels inside the CUB ground-truth bounding box are pasted back into the image.

### Background Mask

Pixels outside the ground-truth bounding box are replaced with gray pixels, while the pixels inside the bounding box remain unchanged.

Therefore, these experiments preserve the rectangular object bounding-box region exactly while modifying its surrounding visual context.

> Note: CUB bounding boxes are used rather than segmentation masks, so these perturbations should not be interpreted as perfectly isolating foreground and background pixels.

---

## 4. Explanation Robustness Metrics

For each original image, the predicted class query from the original image is fixed and used to compare attention before and after perturbation.

Attention is taken from the **final transformer decoder layer** and averaged across its eight attention heads.

We evaluate explanation similarity using:

- **Cosine similarity**
- **Pearson correlation**
- **Top-20% attention IoU**

Our primary criterion considers attention stable when:

```text
Cosine similarity >= 0.90
AND
Top-20% IoU >= 0.50
```

We additionally track prediction consistency and changes in the fixed-query classification margin.

---

## 5. Training Methods

We compare the pretrained INTR model with three fine-tuning variants.

### λ0 — Paired Fine-Tuning Control

The model is fine-tuned using paired original and background-blurred images, but without an attention consistency objective.

```math
L_{\mathrm{cls}}
=
\frac{1}{2}
\left[
\mathrm{CE}(z(x), y)
+
\mathrm{CE}(z(\tilde{x}), y)
\right]
```

This controls for the effect of paired perturbation training itself.

### V1 — Attention Consistency

V1 adds a consistency objective between the class-query attention maps of the original and blurred images:

```math
L_{\mathrm{cons}}
=
1-\cos\left(A_y(x), A_y(\tilde{x})\right)
```

The complete objective is:

```math
L
=
L_{\mathrm{cls}}
+
\lambda L_{\mathrm{cons}}
```

with $\lambda = 0.01$.

### V2 — Teacher-Anchored Attention Consistency

Diagnostics showed that V1 greatly improved perturbation consistency but also increased similarity between attention maps from different images.

V2 therefore introduces a frozen pretrained INTR model as a teacher and anchors the student's clean-image attention to the original model:

```math
L_{\mathrm{anchor}}
=
1-\cos\left(
A_y^{\mathrm{student}}(x),
A_y^{\mathrm{teacher}}(x)
\right)
```

The complete objective is:

```math
L
=
L_{\mathrm{cls}}
+
0.01 L_{\mathrm{cons}}
+
0.0015 L_{\mathrm{anchor}}
```

The teacher is frozen throughout training.
---

## 6. Final Evaluation

All fine-tuning methods were trained using three random seeds:

```text
42, 43, 44
```

Final evaluation was performed on the untouched 5,394-image test cohort.

### Background Blur

| Method | Prediction Stable | Attention Unstable | B / Stable | Cosine | Pearson | Top-20 IoU |
|---|---:|---:|---:|---:|---:|---:|
| Pretrained INTR | 83.5% | 43.3% | 37.3% | 0.8977 | 0.8056 | 0.6255 |
| λ0 | 84.8 ± 0.2% | 41.4 ± 0.1% | 36.5 ± 0.1% | 0.9010 ± 0.0010 | 0.8126 ± 0.0004 | 0.6342 ± 0.0005 |
| V1 | 85.6 ± 0.3% | 6.7 ± 0.2% | 4.7 ± 0.2% | 0.9809 ± 0.0001 | 0.9397 ± 0.0003 | 0.7730 ± 0.0004 |
| V2 | 85.8 ± 0.1% | 6.5 ± 0.2% | 4.6 ± 0.3% | 0.9794 ± 0.0000 | 0.9403 ± 0.0006 | 0.7759 ± 0.0007 |

### Background Mask

| Method | Prediction Stable | Attention Unstable | B / Stable | Cosine | Pearson | Top-20 IoU |
|---|---:|---:|---:|---:|---:|---:|
| Pretrained INTR | 76.3% | 54.1% | 45.8% | 0.8703 | 0.7609 | 0.5809 |
| λ0 | 78.7 ± 0.3% | 52.8 ± 0.0% | 46.0 ± 0.2% | 0.8731 ± 0.0015 | 0.7668 ± 0.0005 | 0.5866 ± 0.0012 |
| V1 | 79.5 ± 0.2% | 12.3 ± 0.1% | 8.4 ± 0.2% | 0.9700 ± 0.0001 | 0.9132 ± 0.0004 | 0.7308 ± 0.0007 |
| V2 | 79.4 ± 0.4% | 12.2 ± 0.1% | 8.3 ± 0.1% | 0.9678 ± 0.0001 | 0.9148 ± 0.0007 | 0.7357 ± 0.0006 |

Results for λ0, V1, and V2 are reported as mean ± standard deviation across the three training seeds.

---

## 7. Attention Homogenization Analysis

Improved perturbation consistency alone does not necessarily imply better explanations. A model could obtain more stable attention simply by producing smoother or more similar attention maps across different images.

We therefore evaluate attention diversity across the final evaluation cohort.

| Method | Normalized Entropy | Max Attention | CV | Cross-Image Cosine | Cross-Image Pearson |
|---|---:|---:|---:|---:|---:|
| λ0 | 0.9247 ± 0.0011 | 0.01332 ± 0.00027 | 1.2688 ± 0.0150 | 0.5373 ± 0.0048 | 0.1884 ± 0.0012 |
| V1 | 0.9495 ± 0.0005 | 0.00515 ± 0.00002 | 0.8598 ± 0.0039 | 0.7655 ± 0.0008 | 0.4605 ± 0.0013 |
| V2 | 0.9438 ± 0.0009 | 0.00567 ± 0.00003 | 0.9285 ± 0.0078 | 0.7416 ± 0.0021 | 0.4455 ± 0.0008 |

V1 substantially improves robustness, but its attention maps become smoother and considerably more similar across different images. This indicates **partial attention homogenization**, rather than complete attention collapse.

The teacher-anchor objective in V2 partially reduces this effect while largely preserving the robustness gains of V1.

---

## 8. Main Findings

The experiments currently support three main observations:

1. **Intrinsic explanations can be unstable even when predictions remain stable.**  
   For pretrained INTR, Category-B cases are common under both background blur and masking.

2. **Attention-consistency training substantially improves explanation robustness.**  
   V1 greatly reduces attention instability under both the training perturbation (blur) and the unseen mask perturbation.

3. **Robustness can introduce a hidden attention-diversity trade-off.**  
   V1 produces substantially more similar attention maps across different images. The teacher-anchored V2 objective partially mitigates this homogenization while retaining most of the robustness improvement.

These results distinguish **explanation robustness** from explanation quality or faithfulness: increased stability alone should not automatically be interpreted as a better explanation.

---

## 9. Repository Structure

```text
intr-research/
├── metadata/
│   ├── formal_400_samples.csv
│   └── final_untouched_5394.csv
│
├── results/
│   ├── development/
│   ├── final_eval/
│   └── final_homogenization/
│
├── src/
│   └── experiment and analysis scripts
│
├── third_party/
│   └── INTR/
│       └── original INTR implementation
│
├── requirements.txt
├── .gitignore
└── README.md
```

### `metadata/`

Contains the frozen sample definitions used to separate method development from final evaluation.

### `results/development/`

Contains pilot experiments and diagnostics used during method development. These results should not be interpreted as the final evaluation.

### `results/final_eval/`

Contains frozen final robustness evaluations for pretrained INTR, λ0, V1, and V2.

### `results/final_homogenization/`

Contains final attention-diversity and homogenization analyses.

### `src/`

Contains the experiment, fine-tuning, robustness evaluation, and analysis scripts developed for this project.

### `third_party/INTR/`

Contains the original INTR implementation used as the research baseline. The original license and citation information are retained in this directory.

---

## 10. Reproducibility

Large datasets and model checkpoints are not included in this repository.

To reproduce the experiments:

1. Download the **CUB-200-2011** dataset.
2. Obtain the pretrained INTR CUB checkpoint.
3. Place the required files according to the paths expected by the experiment scripts.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the relevant scripts in `src/` for fine-tuning, robustness evaluation, and analysis.

The frozen evaluation cohorts are provided in `metadata/`, and the reported experiment outputs are provided in `results/`.

---

## 11. Upstream INTR

This project builds on the official INTR implementation:

**A Simple Interpretable Transformer for Fine-Grained Image Classification and Analysis**  
ICLR 2024

The original implementation, license, and citation information are preserved under:

```text
third_party/INTR/
```

This repository's experimental code and analyses are separate from the upstream INTR implementation.
