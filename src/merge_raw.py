import shutil
from pathlib import Path

RAW = Path("data/raw")
(RAW / "images").mkdir(parents=True, exist_ok=True)
(RAW / "labels").mkdir(parents=True, exist_ok=True)

for split in ["train", "valid"]:
    img_src = RAW / split / "images"
    lbl_src = RAW / split / "labels"

    for img in img_src.glob("*"):
        shutil.copy(img, RAW / "images" / img.name)

    for lbl in lbl_src.glob("*"):
        shutil.copy(lbl, RAW / "labels" / lbl.name)

print("Merged. Total images:", len(list((RAW / "images").glob("*"))))
print("Total labels:", len(list((RAW / "labels").glob("*"))))