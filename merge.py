"""
PitSense Dataset Merger
Converts archive (Pascal VOC XML) → YOLO format
then merges with BharatPothole into a single unified dataset.

Output: C:\Projects\PitSense\merged_dataset\
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
BASE          = Path(r"C:\Projects\PitSense")
ARCHIVE_IMG   = BASE / "archive" / "images"
ARCHIVE_ANN   = BASE / "archive" / "annotations"
BHARAT_TRAIN  = BASE / "BharatPotHole" / "BharatPotHole" / "BharatPotHole" / "train"
BHARAT_VALID  = BASE / "BharatPotHole" / "BharatPotHole" / "BharatPotHole" / "valid"
BHARAT_TEST   = BASE / "BharatPotHole" / "BharatPotHole" / "BharatPotHole" / "test"
OUT           = BASE / "merged_dataset"

# Split ratios for archive dataset (it has no pre-split)
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
# TEST_RATIO  = 0.10  (remainder)

random.seed(42)

# ── Helper: Pascal VOC XML → YOLO txt ─────────────────────────
def voc_to_yolo(xml_path: Path, out_txt: Path):
    """
    Converts a single Pascal VOC XML file to YOLO format.
    Returns True if at least one valid box was written.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size  = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        if img_w == 0 or img_h == 0:
            return False

        lines = []
        for obj in root.findall("object"):
            # Skip difficult / truncated if flagged
            diff = obj.find("difficult")
            if diff is not None and int(diff.text) == 1:
                continue

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Clamp to image bounds
            xmin = max(0, min(xmin, img_w))
            ymin = max(0, min(ymin, img_h))
            xmax = max(0, min(xmax, img_w))
            ymax = max(0, min(ymax, img_h))

            if xmax <= xmin or ymax <= ymin:
                continue

            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h

            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            return False

        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text("\n".join(lines))
        return True

    except Exception as e:
        print(f"  [WARN] Failed to parse {xml_path.name}: {e}")
        return False

# ── Helper: copy image + label into split folder ───────────────
def copy_pair(img_src: Path, lbl_src: Path, split: str, prefix: str = ""):
    img_dst = OUT / split / "images" / f"{prefix}{img_src.name}"
    lbl_dst = OUT / split / "labels" / f"{prefix}{lbl_src.stem}.txt"
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_src, img_dst)
    shutil.copy2(lbl_src, lbl_dst)

# ── Step 1: Convert + split archive dataset ───────────────────
print("=" * 60)
print("Step 1 — Converting archive (VOC XML → YOLO)")
print("=" * 60)

tmp_labels = BASE / "_tmp_archive_labels"
tmp_labels.mkdir(exist_ok=True)

archive_pairs = []   # (img_path, yolo_txt_path)
skipped = 0

xml_files = sorted(ARCHIVE_ANN.glob("*.xml"))
for xml_path in xml_files:
    stem      = xml_path.stem                          # e.g. potholes0
    img_path  = ARCHIVE_IMG / f"{stem}.png"
    yolo_txt  = tmp_labels / f"{stem}.txt"

    if not img_path.exists():
        skipped += 1
        continue

    ok = voc_to_yolo(xml_path, yolo_txt)
    if ok:
        archive_pairs.append((img_path, yolo_txt))
    else:
        skipped += 1

print(f"  Converted : {len(archive_pairs)} images")
print(f"  Skipped   : {skipped} (no valid boxes or missing image)")

# Shuffle and split
random.shuffle(archive_pairs)
n        = len(archive_pairs)
n_train  = int(n * TRAIN_RATIO)
n_valid  = int(n * VALID_RATIO)

splits = {
    "train": archive_pairs[:n_train],
    "valid": archive_pairs[n_train:n_train + n_valid],
    "test":  archive_pairs[n_train + n_valid:],
}

print(f"\n  Archive split:")
for s, pairs in splits.items():
    print(f"    {s:5s}: {len(pairs)} images")

for split, pairs in splits.items():
    for img_path, lbl_path in pairs:
        copy_pair(img_path, lbl_path, split, prefix="arc_")

print("  Done.")

# ── Step 2: Copy BharatPothole dataset ────────────────────────
print("\n" + "=" * 60)
print("Step 2 — Copying BharatPothole dataset")
print("=" * 60)

def copy_bharat_split(src_dir: Path, split: str):
    img_dir = src_dir / "images"
    lbl_dir = src_dir / "labels"
    if not img_dir.exists():
        print(f"  [WARN] {img_dir} not found, skipping.")
        return 0
    count = 0
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        copy_pair(img_path, lbl_path, split, prefix="bh_")
        count += 1
    return count

for split, src in [("train", BHARAT_TRAIN),
                   ("valid", BHARAT_VALID),
                   ("test",  BHARAT_TEST)]:
    n = copy_bharat_split(src, split)
    print(f"  {split:5s}: {n} images copied")

# ── Step 3: Write data.yaml ───────────────────────────────────
print("\n" + "=" * 60)
print("Step 3 — Writing data.yaml")
print("=" * 60)

yaml_content = f"""# PitSense Merged Dataset
# BharatPothole (Indian dashcam) + Roboflow Public Pothole (VOC)

path: {OUT.as_posix()}
train: train/images
val:   valid/images
test:  test/images

nc: 1
names: ['pothole']
"""

yaml_path = OUT / "data.yaml"
yaml_path.write_text(yaml_content)
print(f"  Saved to: {yaml_path}")

# ── Step 4: Final summary ─────────────────────────────────────
print("\n" + "=" * 60)
print("Final dataset summary")
print("=" * 60)

total = 0
for split in ["train", "valid", "test"]:
    img_dir = OUT / split / "images"
    count   = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
    total  += count
    print(f"  {split:5s}: {count} images")

print(f"  TOTAL : {total} images")
print(f"\n  Dataset saved to: {OUT}")
print(f"  data.yaml path  : {yaml_path}")

# Cleanup temp folder
shutil.rmtree(tmp_labels, ignore_errors=True)

print("\nDone! Ready to train.")