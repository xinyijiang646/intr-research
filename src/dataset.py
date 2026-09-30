from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTR_ROOT = PROJECT_ROOT / "third_party" / "INTR"
CUB_ROOT = PROJECT_ROOT / "data" / "CUB_200_2011"

sys.path.insert(0, str(INTR_ROOT))

import datasets.transforms as T


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CUB_ROOT = PROJECT_ROOT / "data" / "CUB_200_2011"


def make_intr_test_transform():
    """
    Official INTR validation/test preprocessing.

    Resize:
        short side -> 800
        long side <= 1333

    Normalize:
        ImageNet mean/std
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def load_cub_metadata(root=CUB_ROOT):
    """
    Load metadata from the official CUB-200-2011 dataset.

    Returns:
    records : list of dict
        One dictionary per image containing:
        image_id, image_path, class_id, class_name,
        is_train, and bounding_box.
    """

    # 1. Image ID -> relative image path
    image_paths = {}

    with open(root / "images.txt", "r") as f:
        for line in f:
            image_id, relative_path = line.strip().split(maxsplit=1)
            image_paths[int(image_id)] = relative_path

    # 2. Image ID -> class ID
    image_labels = {}

    with open(root / "image_class_labels.txt", "r") as f:
        for line in f:
            image_id, class_id = line.strip().split()
            image_labels[int(image_id)] = int(class_id)

    # 3. Image ID -> train/test split
    train_test_split = {}

    with open(root / "train_test_split.txt", "r") as f:
        for line in f:
            image_id, is_train = line.strip().split()
            train_test_split[int(image_id)] = bool(int(is_train))

    # 4. Class ID -> class name
    class_names = {}

    with open(root / "classes.txt", "r") as f:
        for line in f:
            class_id, class_name = line.strip().split(maxsplit=1)
            class_names[int(class_id)] = class_name

    # 5. Image ID -> bounding box
    bounding_boxes = {}

    with open(root / "bounding_boxes.txt", "r") as f:
        for line in f:
            image_id, x, y, width, height = line.strip().split()

            bounding_boxes[int(image_id)] = (
                float(x),
                float(y),
                float(width),
                float(height),
            )

    # Combine all metadata using image_id
    records = []

    for image_id in sorted(image_paths):
        class_id = image_labels[image_id]

        record = {
            "image_id": image_id,
            "image_path": root / "images" / image_paths[image_id],
            "class_id": class_id,
            "class_name": class_names[class_id],
            "is_train": train_test_split[image_id],
            "bounding_box": bounding_boxes[image_id],
        }

        records.append(record)

    return records


def check_image_files(records):
    missing = []

    for record in records:
        if not record["image_path"].exists():
            missing.append(record)

    print()
    print("=== Image File Integrity Check ===")
    print("Expected images:", len(records))
    print("Existing images:", len(records) - len(missing))
    print("Missing images:", len(missing))

    if missing:
        print("\nFirst missing files:")
        for record in missing[:10]:
            print(
                f"Image ID {record['image_id']}: "
                f"{record['image_path']}"
            )

    return missing


class CUBDataset(Dataset):
    """
    CUB-200-2011 dataset using the official train/test split.

    This class keeps the original CUB metadata while allowing
    an INTR-compatible image transform to be applied.
    """

    def __init__(self, root=CUB_ROOT, split="test", transform=None):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        self.root = Path(root)
        self.split = split
        self.transform = transform

        records = load_cub_metadata(self.root)

        if split == "train":
            self.records = [r for r in records if r["is_train"]]
        else:
            self.records = [r for r in records if not r["is_train"]]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image = Image.open(record["image_path"]).convert("RGB")
        original_width, original_height = image.size

        target = {
            "image_id": record["image_id"],
            "class_id": record["class_id"],
            "class_name": record["class_name"],
            "bounding_box": record["bounding_box"],
            "original_size": (original_width, original_height),
        }

        if self.transform is not None:
            image, _ = self.transform(image, None)

        return image, target


if __name__ == "__main__":
    records = load_cub_metadata()
    missing = check_image_files(records)

    train_records = [r for r in records if r["is_train"]]
    test_records = [r for r in records if not r["is_train"]]

    print("=== CUB-200-2011 Metadata Check ===")
    print("Total images:", len(records))
    print("Train images:", len(train_records))
    print("Test images:", len(test_records))
    print()

    transform = make_intr_test_transform()

    dataset = CUBDataset(
        root=CUB_ROOT,
        split="test",
        transform=transform,
    )

    image, target = dataset[0]

    print("=== First Test Image ===")
    print("Dataset size:", len(dataset))
    print("Image tensor shape:", image.shape)
    print("Image dtype:", image.dtype)

    for key, value in target.items():
        print(f"{key}: {value}")
