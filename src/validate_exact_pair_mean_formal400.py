
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

from config import get_intr_config
from model import load_intr_model
from dataset import (
    CUBDataset,
    CUB_ROOT,
    make_intr_test_transform,
)
from robustness import run_intr
from analyze_bbox_attention import (
    load_csv,
    build_dataset_lookup,
)
from check_attention_collapse import (
    resize_attention_map,
    cosine,
    pearson,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FORMAL_SET_PATH = RESULTS_DIR / "formal_400_samples.csv"

CHECKPOINT = Path(
    "/kaggle/input/datasets/xinyijianng/"
    "full-finetune-seed-42-lambda-0p0-pth/"
    "full_finetune_seed_42_lambda_0p0.pth"
)

formal_rows = load_csv(FORMAL_SET_PATH)
assert len(formal_rows) == 400

args = get_intr_config()

if torch.cuda.is_available():
    args.device = "cuda:0"
else:
    args.device = "cpu"

print("device:", args.device)

model = load_intr_model(
    args,
    CHECKPOINT,
)

transform = make_intr_test_transform()

dataset = CUBDataset(
    root=CUB_ROOT,
    split="test",
    transform=None,
)

lookup = build_dataset_lookup(
    dataset
)

maps = []
class_ids = []

for i, row in enumerate(
    formal_rows,
    start=1,
):
    image_id = int(row["image_id"])
    class_id = int(row["class_id"])

    record = lookup[image_id]

    image = Image.open(
        record["image_path"]
    ).convert("RGB")

    image_tensor, _ = transform(
        image,
        None,
    )

    logits, attention, encoder_output = run_intr(
        model,
        image_tensor,
        args.device,
    )

    pred_index = logits.argmax().item()

    fixed_query_attention = (
        attention[pred_index]
    )

    resized = resize_attention_map(
        fixed_query_attention,
        encoder_output,
        output_size=(25, 25),
    )

    maps.append(resized)
    class_ids.append(class_id)

    if i % 100 == 0:
        print(f"{i}/400")


maps = torch.stack(maps).float()
class_ids = torch.tensor(class_ids)


# OLD explicit pair enumeration
old_cos_all = []
old_pear_all = []

old_cos_same = []
old_pear_same = []

old_cos_diff = []
old_pear_diff = []

n = len(maps)

for i in range(n):
    for j in range(i + 1, n):

        c = cosine(
            maps[i],
            maps[j],
        )

        p = pearson(
            maps[i],
            maps[j],
        )

        old_cos_all.append(c)
        old_pear_all.append(p)

        if class_ids[i] == class_ids[j]:
            old_cos_same.append(c)
            old_pear_same.append(p)
        else:
            old_cos_diff.append(c)
            old_pear_diff.append(p)


# NEW exact vectorized pair mean
def exact_pair_sum(unit_vectors):
    n_vec = unit_vectors.shape[0]

    summed = unit_vectors.sum(
        dim=0
    )

    pair_sum = (
        torch.dot(
            summed,
            summed,
        )
        - n_vec
    ) / 2.0

    n_pairs = (
        n_vec * (n_vec - 1) // 2
    )

    return (
        pair_sum.item(),
        n_pairs,
    )


# cosine
cos_vecs = F.normalize(
    maps,
    p=2,
    dim=1,
)

cos_total_sum, n_all = exact_pair_sum(
    cos_vecs
)

cos_same_sum = 0.0
n_same = 0

for c in torch.unique(class_ids):

    group = cos_vecs[
        class_ids == c
    ]

    if len(group) < 2:
        continue

    s, k = exact_pair_sum(group)

    cos_same_sum += s
    n_same += k

cos_diff_sum = (
    cos_total_sum
    - cos_same_sum
)

n_diff = (
    n_all
    - n_same
)


# pearson
centered = (
    maps
    - maps.mean(
        dim=1,
        keepdim=True,
    )
)

pear_vecs = F.normalize(
    centered,
    p=2,
    dim=1,
)

pear_total_sum, _ = exact_pair_sum(
    pear_vecs
)

pear_same_sum = 0.0

for c in torch.unique(class_ids):

    group = pear_vecs[
        class_ids == c
    ]

    if len(group) < 2:
        continue

    s, _ = exact_pair_sum(group)

    pear_same_sum += s

pear_diff_sum = (
    pear_total_sum
    - pear_same_sum
)


# Compare
def m(x):
    return sum(x) / len(x)

results = [
    (
        "cosine all",
        m(old_cos_all),
        cos_total_sum / n_all,
    ),
    (
        "cosine same",
        m(old_cos_same),
        cos_same_sum / n_same,
    ),
    (
        "cosine diff",
        m(old_cos_diff),
        cos_diff_sum / n_diff,
    ),
    (
        "pearson all",
        m(old_pear_all),
        pear_total_sum / n_all,
    ),
    (
        "pearson same",
        m(old_pear_same),
        pear_same_sum / n_same,
    ),
    (
        "pearson diff",
        m(old_pear_diff),
        pear_diff_sum / n_diff,
    ),
]

print()
print("=== Exact Mean Validation ===")

for name, old, new in results:
    print(
        f"{name:16s} "
        f"old={old:.9f} "
        f"new={new:.9f} "
        f"diff={abs(old-new):.12f}"
    )

print()
print(
    "pair counts:",
    n_all,
    n_same,
    n_diff,
)
