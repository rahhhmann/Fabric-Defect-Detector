"""
oversample_seam.py
-------------------------------------------------------------------
Purpose : Fix "seam" class underperformance (AP50=0.601) by oversampling
          train images that contain the "seam" class (class index 4),
          per Option 1 in the seam-weakness brief.

Strategy: Non-destructive. The original processed/images/train and
          processed/labels/train folders are NEVER modified. Instead,
          a NEW folder (oversampled/) is created containing:
            - symlinks/copies of ALL original train images+labels
            - N extra duplicate copies of every image+label pair that
              contains at least one "seam" (class 4) box
          A new dataset_oversampled.yaml points to this folder for
          train, while val/test continue pointing at the original
          unmodified processed/ paths.

Usage (on Kaggle):
    python oversample_seam.py \
        --data-root /kaggle/input/datasets/singertv/fabric-processed/processed \
        --out-root  /kaggle/working/oversampled \
        --yaml-out  /kaggle/working/dataset_oversampled.yaml \
        --seam-class-id 4 \
        --multiplier 3

After running, point src/train.py at:
    /kaggle/working/dataset_oversampled.yaml
instead of the original dataset.yaml.
-------------------------------------------------------------------
"""

import argparse
import shutil
from pathlib import Path

CLASS_NAMES = ["Stain", "Thread", "Warp_Weft", "hole", "seam"]


def find_seam_images(labels_dir: Path, seam_class_id: int):
    """Return sorted list of label file paths (stem) that contain >=1 seam box."""
    seam_stems = []
    all_stems = []
    for lbl_file in sorted(labels_dir.glob("*.txt")):
        all_stems.append(lbl_file.stem)
        has_seam = False
        text = lbl_file.read_text().strip()
        if not text:
            continue  # background-only (no objects) image, skip
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            cls_id = int(line.split()[0])
            if cls_id == seam_class_id:
                has_seam = True
                break
        if has_seam:
            seam_stems.append(lbl_file.stem)
    return all_stems, seam_stems


def copy_pair(img_src: Path, lbl_src: Path, img_dst: Path, lbl_dst: Path, use_symlink: bool):
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    if use_symlink:
        if img_dst.exists() or img_dst.is_symlink():
            img_dst.unlink()
        if lbl_dst.exists() or lbl_dst.is_symlink():
            lbl_dst.unlink()
        img_dst.symlink_to(img_src.resolve())
        lbl_dst.symlink_to(lbl_src.resolve())
    else:
        shutil.copy2(img_src, img_dst)
        shutil.copy2(lbl_src, lbl_dst)


def find_image_for_stem(images_dir: Path, stem: str):
    """Images may be .jpg/.jpeg/.png — find whichever extension exists."""
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = list(images_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def main():
    ap = argparse.ArgumentParser(description="Oversample seam-class train images.")
    ap.add_argument("--data-root", required=True,
                     help="Path to processed/ folder containing images/ and labels/ subfolders "
                          "with train/val/test splits.")
    ap.add_argument("--out-root", required=True,
                     help="Output folder where the oversampled train split will be written "
                          "(val/test are referenced from the original data-root, untouched).")
    ap.add_argument("--yaml-out", required=True,
                     help="Path to write the new dataset_oversampled.yaml")
    ap.add_argument("--seam-class-id", type=int, default=4,
                     help="Class index for 'seam' (default 4, per CLASS_NAMES order).")
    ap.add_argument("--multiplier", type=int, default=3,
                     help="How many EXTRA duplicate copies to add per seam image "
                          "(3 means each seam image appears 4x total in the new train set).")
    ap.add_argument("--use-symlink", action="store_true", default=True,
                     help="Use symlinks instead of full file copies (saves disk space). "
                          "Falls back to copy automatically if symlinks aren't supported.")
    ap.add_argument("--no-symlink", dest="use_symlink", action="store_false",
                     help="Force real file copies instead of symlinks.")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    images_train = data_root / "images" / "train"
    labels_train = data_root / "labels" / "train"

    if not images_train.exists() or not labels_train.exists():
        raise SystemExit(f"ERROR: expected {images_train} and {labels_train} to exist.")

    out_images_train = out_root / "images" / "train"
    out_labels_train = out_root / "labels" / "train"

    print(f"[1/4] Scanning labels in {labels_train} ...")
    all_stems, seam_stems = find_seam_images(labels_train, args.seam_class_id)
    print(f"      Total train label files : {len(all_stems)}")
    print(f"      Seam-containing images   : {len(seam_stems)}")

    if len(seam_stems) == 0:
        raise SystemExit("ERROR: found 0 seam-containing images. Check --seam-class-id "
                          "matches the class index for 'seam' in your data.yaml (classes "
                          f"assumed order: {CLASS_NAMES}).")

    # quick symlink support probe
    use_symlink = args.use_symlink
    if use_symlink:
        try:
            probe_src = images_train.iterdir().__next__()
            probe_dst = out_root / "_symlink_probe"
            out_root.mkdir(parents=True, exist_ok=True)
            probe_dst.symlink_to(probe_src.resolve())
            probe_dst.unlink()
        except Exception as e:
            print(f"      Symlinks not supported here ({e}); falling back to file copies.")
            use_symlink = False

    print(f"[2/4] Writing base train set (all {len(all_stems)} original pairs) "
          f"to {out_root} via {'symlinks' if use_symlink else 'copies'} ...")
    missing = 0
    for stem in all_stems:
        img_src = find_image_for_stem(images_train, stem)
        lbl_src = labels_train / f"{stem}.txt"
        if img_src is None or not lbl_src.exists():
            missing += 1
            continue
        img_dst = out_images_train / img_src.name
        lbl_dst = out_labels_train / f"{stem}.txt"
        copy_pair(img_src, lbl_src, img_dst, lbl_dst, use_symlink)
    if missing:
        print(f"      WARNING: {missing} label files had no matching image, skipped.")

    print(f"[3/4] Adding {args.multiplier}x extra duplicate(s) for each of the "
          f"{len(seam_stems)} seam images ...")
    added = 0
    for stem in seam_stems:
        img_src = find_image_for_stem(images_train, stem)
        lbl_src = labels_train / f"{stem}.txt"
        if img_src is None:
            continue
        ext = img_src.suffix
        for k in range(1, args.multiplier + 1):
            new_stem = f"{stem}_seamdup{k}"
            img_dst = out_images_train / f"{new_stem}{ext}"
            lbl_dst = out_labels_train / f"{new_stem}.txt"
            copy_pair(img_src, lbl_src, img_dst, lbl_dst, use_symlink)
            added += 1

    final_count = len(all_stems) + added
    print(f"      Added {added} duplicate image/label pairs.")
    print(f"      Final oversampled train set size: {final_count} images "
          f"(was {len(all_stems)})")

    print(f"[4/4] Writing {args.yaml_out} ...")
    # Use absolute paths everywhere and omit the ambiguous top-level `path:` key
    # (Ultralytics joins `path` + train/val/test, which causes errors here since
    # the oversampled train dir and original val/test dirs don't share one parent).
    yaml_content = f"""# Auto-generated by oversample_seam.py
# Train points to the oversampled set; val/test stay on the original, untouched data.
train: {out_images_train.resolve()}
val: {(data_root / 'images' / 'val').resolve()}
test: {(data_root / 'images' / 'test').resolve()}

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    Path(args.yaml_out).write_text(yaml_content)
    print("\nDONE.")
    print(f"  Original train images : {len(all_stems)}")
    print(f"  Seam images found     : {len(seam_stems)}")
    print(f"  Duplicates added      : {added}")
    print(f"  New train set size    : {final_count}")
    print(f"  New yaml              : {args.yaml_out}")
    print("\nNext step: point src/train.py's data= argument at this new yaml file.")


if __name__ == "__main__":
    main()
