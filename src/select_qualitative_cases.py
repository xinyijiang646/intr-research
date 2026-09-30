import csv
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

BLUR_PATH = (
    RESULTS_DIR
    / "formal_400_background_blur.csv"
)

MASK_PATH = (
    RESULTS_DIR
    / "formal_400_background_mask.csv"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "qualitative_cases.csv"
)


def load_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def build_by_id(rows):
    return {
        int(row["image_id"]): row
        for row in rows
    }


def get_float(row, key):
    return float(row[key])


def attention_instability_score(
    blur_row,
    mask_row,
):
    """
    Larger score = stronger attention change.

    Uses both cosine and top-20% IoU
    from both perturbations.

    Score:
        (1 - blur cosine)
      + (1 - mask cosine)
      + (1 - blur IoU)
      + (1 - mask IoU)
    """

    blur_cosine = get_float(
        blur_row,
        "attention_cosine",
    )

    mask_cosine = get_float(
        mask_row,
        "attention_cosine",
    )

    blur_iou = get_float(
        blur_row,
        "attention_top20_iou",
    )

    mask_iou = get_float(
        mask_row,
        "attention_top20_iou",
    )

    return (
        (1.0 - blur_cosine)
        + (1.0 - mask_cosine)
        + (1.0 - blur_iou)
        + (1.0 - mask_iou)
    )


def build_case_record(
    case_type,
    image_id,
    blur_row,
    mask_row,
    score,
):
    return {
        "case_type":
            case_type,

        "image_id":
            image_id,

        "class_id":
            int(blur_row["class_id"]),

        "class_name":
            blur_row["class_name"],

        "blur_category":
            blur_row["category"],

        "mask_category":
            mask_row["category"],

        "blur_prediction_unchanged":
            blur_row[
                "prediction_unchanged"
            ],

        "mask_prediction_unchanged":
            mask_row[
                "prediction_unchanged"
            ],

        "blur_cosine":
            get_float(
                blur_row,
                "attention_cosine",
            ),

        "mask_cosine":
            get_float(
                mask_row,
                "attention_cosine",
            ),

        "blur_iou":
            get_float(
                blur_row,
                "attention_top20_iou",
            ),

        "mask_iou":
            get_float(
                mask_row,
                "attention_top20_iou",
            ),

        "blur_pearson":
            get_float(
                blur_row,
                "attention_pearson",
            ),

        "mask_pearson":
            get_float(
                mask_row,
                "attention_pearson",
            ),

        "blur_margin_change":
            get_float(
                blur_row,
                "margin_change",
            ),

        "mask_margin_change":
            get_float(
                mask_row,
                "margin_change",
            ),

        "attention_instability_score":
            score,
    }


if __name__ == "__main__":

    blur_rows = load_csv(
        BLUR_PATH
    )

    mask_rows = load_csv(
        MASK_PATH
    )

    blur_by_id = build_by_id(
        blur_rows
    )

    mask_by_id = build_by_id(
        mask_rows
    )

    blur_ids = set(
        blur_by_id.keys()
    )

    mask_ids = set(
        mask_by_id.keys()
    )

    print(
        "=== Qualitative Case Selection ==="
    )

    print(
        "Blur images:",
        len(blur_ids),
    )

    print(
        "Mask images:",
        len(mask_ids),
    )

    print(
        "Same image set:",
        blur_ids == mask_ids,
    )

    assert blur_ids == mask_ids

    # Persistent B candidates
    persistent_b = []

    for image_id in sorted(
        blur_ids
    ):

        blur_row = blur_by_id[
            image_id
        ]

        mask_row = mask_by_id[
            image_id
        ]

        if (
            blur_row["category"] == "B"
            and
            mask_row["category"] == "B"
        ):
            score = (
                attention_instability_score(
                    blur_row,
                    mask_row,
                )
            )

            persistent_b.append(
                (
                    image_id,
                    score,
                )
            )

    print()
    print(
        "Persistent B candidates:",
        len(persistent_b),
    )

    assert len(
        persistent_b
    ) > 0

    # Sort from weakest to strongest
    persistent_b.sort(
        key=lambda item: item[1]
    )

    # Strong B
    # highest instability score
    strong_b_id, strong_b_score = (
        persistent_b[-1]
    )

    # Mild B
    # lowest instability score while
    # still meeting B under both
    mild_b_id, mild_b_score = (
        persistent_b[0]
    )

    # Moderate B
    # closest to median instability score
    b_scores = [
        score
        for _, score
        in persistent_b
    ]

    median_b_score = median(
        b_scores
    )

    moderate_b_id, moderate_b_score = min(
        persistent_b,
        key=lambda item:
            abs(
                item[1]
                - median_b_score
            ),
    )

    # Persistent A candidates
    persistent_a = []

    for image_id in sorted(
        blur_ids
    ):

        blur_row = blur_by_id[
            image_id
        ]

        mask_row = mask_by_id[
            image_id
        ]

        if (
            blur_row["category"] == "A"
            and
            mask_row["category"] == "A"
        ):
            score = (
                attention_instability_score(
                    blur_row,
                    mask_row,
                )
            )

            persistent_a.append(
                (
                    image_id,
                    score,
                )
            )

    print(
        "Persistent A candidates:",
        len(persistent_a),
    )

    assert len(
        persistent_a
    ) > 0

    # Choose a representative A,
    # not the most perfectly stable one.
    a_scores = [
        score
        for _, score
        in persistent_a
    ]

    median_a_score = median(
        a_scores
    )

    control_a_id, control_a_score = min(
        persistent_a,
        key=lambda item:
            abs(
                item[1]
                - median_a_score
            ),
    )

    # Build final selected cases
    selections = [
        (
            "strong_persistent_B",
            strong_b_id,
            strong_b_score,
        ),
        (
            "moderate_persistent_B",
            moderate_b_id,
            moderate_b_score,
        ),
        (
            "mild_persistent_B",
            mild_b_id,
            mild_b_score,
        ),
        (
            "persistent_A_control",
            control_a_id,
            control_a_score,
        ),
    ]

    selected_rows = []

    for (
        case_type,
        image_id,
        score,
    ) in selections:

        blur_row = blur_by_id[
            image_id
        ]

        mask_row = mask_by_id[
            image_id
        ]

        selected_rows.append(
            build_case_record(
                case_type,
                image_id,
                blur_row,
                mask_row,
                score,
            )
        )

    # Print selected cases
    print()
    print(
        "=== Selected Qualitative Cases ==="
    )

    for row in selected_rows:

        print()
        print(
            row["case_type"]
        )

        print(
            f"  Image ID: "
            f"{row['image_id']}"
        )

        print(
            f"  Class: "
            f"{row['class_id']} | "
            f"{row['class_name']}"
        )

        print(
            f"  Categories: "
            f"Blur "
            f"{row['blur_category']} | "
            f"Mask "
            f"{row['mask_category']}"
        )

        print(
            f"  Blur: "
            f"Cos "
            f"{row['blur_cosine']:.3f} | "
            f"IoU "
            f"{row['blur_iou']:.3f} | "
            f"Pearson "
            f"{row['blur_pearson']:.3f}"
        )

        print(
            f"  Mask: "
            f"Cos "
            f"{row['mask_cosine']:.3f} | "
            f"IoU "
            f"{row['mask_iou']:.3f} | "
            f"Pearson "
            f"{row['mask_pearson']:.3f}"
        )

        print(
            f"  Score: "
            f"{row['attention_instability_score']:.3f}"
        )

    # Save
    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                selected_rows[0].keys(),
        )

        writer.writeheader()

        writer.writerows(
            selected_rows
        )

    print()
    print(
        "Saved qualitative case list to:"
    )

    print(
        OUTPUT_PATH
    )