"""Convert KITTI object-detection labels to YOLO format with train/val split."""

from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2

# KITTI class name -> YOLO class id (Van/Truck/DontCare/Misc ignored)
CLASS_MAP: dict[str, int] = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
}

IGNORE_TYPES = frozenset({"DontCare", "Misc", "Van", "Truck", "Tram", "Person_sitting"})


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_kitti_line(line: str) -> tuple[str, float, float, float, float] | None:
    """Return (class_name, x1, y1, x2, y2) or None if skipped."""
    parts = line.strip().split()
    if len(parts) < 15:
        return None
    obj_type = parts[0]
    if obj_type in IGNORE_TYPES or obj_type not in CLASS_MAP:
        return None
    x1, y1, x2, y2 = map(float, parts[4:8])
    return obj_type, x1, y1, x2, y2


def to_yolo_line(class_id: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    """Convert pixel bbox to normalized YOLO line."""
    bw = max(x2 - x1, 0.0)
    bh = max(y2 - y1, 0.0)
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    nw = bw / w
    nh = bh / h
    # Clamp to [0, 1] for safety
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def link_or_copy(src: Path, dst: Path) -> None:
    """Symlink image when possible; fall back to copy on Windows without privileges."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert_split(
    stems: list[str],
    split_name: str,
    kitti_img_dir: Path,
    kitti_lbl_dir: Path,
    out_root: Path,
    class_counter: Counter,
) -> int:
    """Write YOLO labels and link/copy images for one split."""
    img_out = out_root / "images" / split_name
    lbl_out = out_root / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    written = 0
    for stem in stems:
        src_img = kitti_img_dir / f"{stem}.png"
        src_lbl = kitti_lbl_dir / f"{stem}.txt"
        if not src_img.is_file():
            continue

        frame = cv2.imread(str(src_img))
        if frame is None:
            print(f"Warning: cannot read image {src_img}")
            continue
        h, w = frame.shape[:2]

        yolo_lines: list[str] = []
        if src_lbl.is_file():
            with open(src_lbl, encoding="utf-8") as f:
                for line in f:
                    parsed = parse_kitti_line(line)
                    if parsed is None:
                        continue
                    cls_name, x1, y1, x2, y2 = parsed
                    class_id = CLASS_MAP[cls_name]
                    yolo_lines.append(to_yolo_line(class_id, x1, y1, x2, y2, w, h))
                    class_counter[cls_name] += 1

        with open(lbl_out / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))
            if yolo_lines:
                f.write("\n")

        link_or_copy(src_img, img_out / f"{stem}.png")
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert KITTI labels to YOLO format")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    root = project_root()
    kitti_root = root / "data" / "kitti" / "training"
    kitti_img_dir = kitti_root / "image_2"
    kitti_lbl_dir = kitti_root / "label_2"
    out_root = root / "data" / "kitti_yolo"

    if not kitti_img_dir.is_dir():
        raise FileNotFoundError(f"Missing KITTI images: {kitti_img_dir}")
    if not kitti_lbl_dir.is_dir():
        raise FileNotFoundError(f"Missing KITTI labels: {kitti_lbl_dir}")

    stems = sorted(p.stem for p in kitti_img_dir.glob("*.png"))
    random.seed(args.seed)
    shuffled = stems.copy()
    random.shuffle(shuffled)

    n_val = int(len(shuffled) * args.val_ratio)
    val_stems = shuffled[:n_val]
    train_stems = shuffled[n_val:]

    # Clean output dirs for idempotent re-runs
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    class_counter: Counter = Counter()
    n_train = convert_split(train_stems, "train", kitti_img_dir, kitti_lbl_dir, out_root, class_counter)
    n_val = convert_split(val_stems, "val", kitti_img_dir, kitti_lbl_dir, out_root, class_counter)

    total_objects = sum(class_counter.values())
    print("=== KITTI to YOLO conversion complete ===")
    print(f"Output directory: {out_root}")
    print(f"Total images:     {len(stems)}")
    print(f"Train images:     {n_train}")
    print(f"Val images:       {n_val}")
    print(f"Total objects:    {total_objects}")
    print("\nClass distribution (included labels):")
    for name in ("Car", "Pedestrian", "Cyclist"):
        print(f"  {name:12s}: {class_counter.get(name, 0)}")


if __name__ == "__main__":
    main()
