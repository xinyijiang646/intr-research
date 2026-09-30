from pathlib import Path
from statistics import mean, median

from PIL import Image

from dataset import (
    CUBDataset,
    CUB_ROOT,
)


def percentile(
    values,
    p,
):
    values = sorted(values)

    index = (
        (len(values) - 1)
        * p
    )

    lower = int(index)

    upper = min(
        lower + 1,
        len(values) - 1,
    )

    weight = (
        index - lower
    )

    return (
        values[lower]
        * (1.0 - weight)
        +
        values[upper]
        * weight
    )


def bbox_area_ratio(
    image_path,
    bounding_box,
):
    with Image.open(
        image_path
    ) as image:
        width, height = (
            image.size
        )

    x, y, bbox_w, bbox_h = (
        bounding_box
    )

    bbox_area = (
        float(bbox_w)
        * float(bbox_h)
    )

    image_area = (
        float(width)
        * float(height)
    )

    return (
        bbox_area
        /
        image_area
    )


def raw_weight(
    scale,
    tau,
    alpha,
):
    smallness = max(
        0.0,
        1.0
        -
        scale / tau,
    )

    return (
        1.0
        +
        alpha
        * smallness
    )


if __name__ == "__main__":

    dataset = CUBDataset(
        root=CUB_ROOT,
        split="train",
        transform=None,
    )

    scales = []

    for record in (
        dataset.records
    ):
        scale = bbox_area_ratio(
            record["image_path"],
            record["bounding_box"],
        )

        scales.append(
            scale
        )

    assert len(scales) == 5994

    q1 = percentile(
        scales,
        0.25,
    )

    q2 = percentile(
        scales,
        0.50,
    )

    q3 = percentile(
        scales,
        0.75,
    )

    print(
        "=== Train bbox scale distribution ==="
    )

    print(
        "N:",
        len(scales),
    )

    print(
        "Min:",
        f"{min(scales):.6f}",
    )

    print(
        "Q1:",
        f"{q1:.6f}",
    )

    print(
        "Median:",
        f"{q2:.6f}",
    )

    print(
        "Q3:",
        f"{q3:.6f}",
    )

    print(
        "Max:",
        f"{max(scales):.6f}",
    )

    print(
        "Mean:",
        f"{mean(scales):.6f}",
    )

    tau = q2

    print()
    print(
        "Reference tau "
        "(train median):",
        f"{tau:.6f}",
    )

    for alpha in [
        0.5,
        1.0,
        2.0,
    ]:
        raw_weights = [
            raw_weight(
                scale,
                tau,
                alpha,
            )
            for scale in scales
        ]

        normalization = mean(
            raw_weights
        )

        normalized = [
            weight
            /
            normalization
            for weight in
            raw_weights
        ]

        print()
        print(
            "=" * 60
        )

        print(
            "alpha =",
            alpha,
        )

        print(
            "Raw weight"
        )

        print(
            "  min:",
            f"{min(raw_weights):.6f}",
        )

        print(
            "  median:",
            f"{median(raw_weights):.6f}",
        )

        print(
            "  max:",
            f"{max(raw_weights):.6f}",
        )

        print(
            "  mean:",
            f"{normalization:.6f}",
        )

        print(
            "Normalized weight"
        )

        print(
            "  min:",
            f"{min(normalized):.6f}",
        )

        print(
            "  median:",
            f"{median(normalized):.6f}",
        )

        print(
            "  max:",
            f"{max(normalized):.6f}",
        )

        print(
            "  mean:",
            f"{mean(normalized):.6f}",
        )