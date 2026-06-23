import os
import shutil
import random
import cv2
import albumentations as A
from pathlib import Path

# ── Config ──────────────────────────────────────────────
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
SPLIT = (0.7, 0.2, 0.1)  # train / val / test
IMG_SIZE = 640
SEED = 42
# ────────────────────────────────────────────────────────

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.3),
    A.Rotate(limit=15, p=0.4),
    A.CLAHE(p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def get_label_path(img_path: str) -> str:
    """OS-independent: navigates from .../images/x.jpg → .../labels/x.txt"""
    p = Path(img_path)
    return str(p.parent.parent / "labels" / (p.stem + ".txt"))


def split_dataset(image_paths, split):
    random.seed(SEED)
    random.shuffle(image_paths)
    n = len(image_paths)
    train_end = int(n * split[0])
    val_end = train_end + int(n * split[1])
    return image_paths[:train_end], image_paths[train_end:val_end], image_paths[val_end:]


def prepare_dirs():
    for split in ['train', 'val', 'test']:
        Path(f"{OUT_DIR}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{OUT_DIR}/labels/{split}").mkdir(parents=True, exist_ok=True)


def process_and_copy(image_paths, split_name, augment_train=False):
    skipped = 0
    corrupted_boxes = 0

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  Could not read image: {img_path}")
            skipped += 1
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        label_path = get_label_path(img_path)
        dst_img = f"{OUT_DIR}/images/{split_name}/{Path(img_path).name}"
        dst_lbl = f"{OUT_DIR}/labels/{split_name}/{Path(label_path).name}"

        cv2.imwrite(dst_img, img)

        if not os.path.exists(label_path):
            print(f"⚠️  No label found for: {img_path}")
            skipped += 1
            continue

        shutil.copy(label_path, dst_lbl)

        # ── Augmentation (train only) ────────────────────────────
        if augment_train and split_name == 'train':
            with open(label_path) as f:
                lines = [l.strip() for l in f.read().strip().split('\n') if l.strip()]

            if not lines:
                continue

            raw_bboxes = []
            raw_labels = []
            for l in lines:
                parts = l.split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                coords = list(map(float, parts[1:]))
                raw_bboxes.append(coords)
                raw_labels.append(cls)

            # Filter out invalid bboxes (any coordinate outside [0.0, 1.0])
            bboxes, labels = [], []
            for bbox, cls in zip(raw_bboxes, raw_labels):
                if all(0.0 <= v <= 1.0 for v in bbox):
                    bboxes.append(bbox)
                    labels.append(cls)
                else:
                    corrupted_boxes += 1

            aug_img_path = dst_img.replace('.jpg', '_aug.jpg').replace('.png', '_aug.png')
            aug_lbl_path = dst_lbl.replace('.txt', '_aug.txt')

            if not bboxes:
                # No valid boxes — save augmented image with empty label
                cv2.imwrite(aug_img_path, img)
                open(aug_lbl_path, 'w').close()
                continue

            try:
                augmented = augment(image=img, bboxes=bboxes, class_labels=labels)
            except Exception as e:
                print(f"⚠️  Augmentation failed for {img_path}: {e}")
                continue

            cv2.imwrite(aug_img_path, augmented['image'])
            with open(aug_lbl_path, 'w') as f:
                for cls, bbox in zip(augmented['class_labels'], augmented['bboxes']):
                    f.write(f"{cls} {' '.join(map(str, bbox))}\n")

    if skipped:
        print(f"  ⚠️  Skipped {skipped} files (missing image or label)")
    if corrupted_boxes:
        print(f"  ⚠️  Filtered {corrupted_boxes} corrupted bounding boxes")


if __name__ == "__main__":
    prepare_dirs()

    all_images = list(Path(f"{RAW_DIR}/images").glob("*.jpg")) + \
                 list(Path(f"{RAW_DIR}/images").glob("*.png"))
    all_images = [str(p) for p in all_images]

    if not all_images:
        print("❌ No images found in data/raw/images/ — check merge_raw.py ran correctly.")
        exit(1)

    train, val, test = split_dataset(all_images, SPLIT)

    print(f"Processing train ({len(train)} images + augmented copies)...")
    process_and_copy(train, 'train', augment_train=True)

    print(f"Processing val ({len(val)} images)...")
    process_and_copy(val, 'val')

    print(f"Processing test ({len(test)} images)...")
    process_and_copy(test, 'test')

    print(f"\n✅ Done! Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")