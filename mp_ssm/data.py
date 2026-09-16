"""Paired segmentation data, synchronized augmentations and video-group splits."""

import csv
import json
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def file_index(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")
    indexed = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in indexed:
                raise ValueError(f"Duplicate filename stem: {path.stem} in {directory}")
            indexed[path.stem] = path
    return indexed


def read_mask(path):
    with Image.open(path) as image:
        mask = np.asarray(image)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be single-channel: {path}")
    values = set(np.unique(mask).tolist())
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError(f"Expected mask values 0/1 or 0/255, found {sorted(values)[:12]}: {path}")
    return (mask > 0).astype(np.float32)


def preprocess_rgb(rgb, settings):
    if settings["clahe"]:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=float(settings["clahe_clip"]),
                               tileGridSize=(int(settings["clahe_grid"]),) * 2)
        lab[..., 0] = clahe.apply(lab[..., 0])
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb


class SegmentationDataset(Dataset):
    def __init__(self, root, split, settings, augment=False):
        self.settings, self.augment = dict(settings), augment
        self.size = int(settings["image_size"])
        root = Path(root) / split
        images, masks = file_index(root / "images"), file_index(root / "masks")
        if not images or images.keys() != masks.keys():
            raise ValueError(f"Invalid {split} pairs: missing masks={sorted(images.keys() - masks.keys())[:5]}, "
                             f"missing images={sorted(masks.keys() - images.keys())[:5]}; images={len(images)}")
        self.pairs = [(images[name], masks[name]) for name in sorted(images)]
        for image_path, mask_path in self.pairs:
            with Image.open(image_path) as image:
                shape = (image.height, image.width)
            mask = read_mask(mask_path)
            if mask.shape != shape:
                raise ValueError(f"Image/mask dimensions differ: {image_path}, {mask_path}")

    def __len__(self):
        return len(self.pairs)

    def load_pair(self, index):
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as image:
            rgb = np.array(image.convert("RGB"))
        rgb = preprocess_rgb(rgb, self.settings)
        mask = read_mask(mask_path)
        size = (self.size, self.size)
        return cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0, \
            cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

    def augment_pair(self, index):
        image, mask = self.load_pair(index)
        s, size = self.settings, self.size
        if random.random() < s["mosaic_p"]:
            cut = size // 2
            quadrants = [(0, cut, 0, cut), (0, cut, cut, size),
                         (cut, size, 0, cut), (cut, size, cut, size)]
            sources = [index] + random.choices(range(len(self)), k=3)
            mosaic_image, mosaic_mask = np.zeros_like(image), np.zeros_like(mask)
            for source, (y0, y1, x0, x1) in zip(sources, quadrants):
                src_image, src_mask = self.load_pair(source)
                mosaic_image[y0:y1, x0:x1] = cv2.resize(src_image, (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
                mosaic_mask[y0:y1, x0:x1] = cv2.resize(src_mask, (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST)
            image, mask = mosaic_image, mosaic_mask
        if random.random() < s["hflip_p"]:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if random.random() < s["vflip_p"]:
            image, mask = image[::-1], mask[::-1]
        if random.random() < s["rotate_p"]:
            angle = random.uniform(-s["rotate_degrees"], s["rotate_degrees"])
            matrix = cv2.getRotationMatrix2D(((size-1)/2, (size-1)/2), angle, 1)
            image = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mask = cv2.warpAffine(mask, matrix, (size, size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        return image.copy(), mask.copy()

    def __getitem__(self, index):
        image, mask = self.augment_pair(index) if self.augment else self.load_pair(index)
        if self.augment and random.random() < self.settings["mixup_p"]:
            other_image, other_mask = self.augment_pair(random.randrange(len(self)))
            alpha = self.settings["mixup_alpha"]
            weight = float(np.random.beta(alpha, alpha))
            image = weight * image + (1-weight) * other_image
            mask = weight * mask + (1-weight) * other_mask
        return {"image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float(),
                "mask": torch.from_numpy(np.ascontiguousarray(mask[None])).float(),
                "name": self.pairs[index][0].stem}


def audit_manifest(root):
    """Require full coverage and video isolation when provenance is supplied."""
    root = Path(root)
    path = root / "manifest.csv"
    if not path.exists():
        return {"video_isolation": "unverified: no manifest.csv", "synthetic": (root / "SYNTHETIC_ONLY.json").exists()}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "mask", "video_id", "split"} <= set(reader.fieldnames or []):
            raise ValueError("manifest.csv needs image,mask,video_id,split")
        rows = list(reader)
    seen_videos, listed = {}, set()
    for row in rows:
        video, split = row["video_id"].strip(), row["split"]
        if not video or split not in {"train", "val", "test"}:
            raise ValueError("Manifest video_id/split is missing or invalid")
        if video in seen_videos and seen_videos[video] != split:
            raise ValueError(f"Video leakage across splits: {video}")
        seen_videos[video] = split
        image, mask = Path(row["image"]), Path(row["mask"])
        for item, folder in ((image, "images"), (mask, "masks")):
            if item.is_absolute() or ".." in item.parts or item.parent != Path(split) / folder:
                raise ValueError(f"Invalid manifest path: {item}")
            if not (root / item).is_file():
                raise ValueError(f"Missing manifest file: {item}")
        if image.stem != mask.stem or str(image) in listed:
            raise ValueError("Manifest has duplicate or mismatched pairs")
        listed.add(str(image))
    actual = {str(p.relative_to(root)) for split in ("train", "val", "test")
              for p in file_index(root / split / "images").values()}
    if listed != actual:
        raise ValueError("manifest.csv must cover every image exactly once")
    return {"video_isolation": "verified from supplied video_id metadata", "videos": len(seen_videos),
            "images": len(listed), "synthetic": (root / "SYNTHETIC_ONLY.json").exists()}


def split_by_video(source, manifest_path, output, seed=42):
    """Copy pairs according to whole video groups, targeting image counts 8:1:1."""
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Split output must be new or empty")
    with Path(manifest_path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "mask", "video_id"} <= set(reader.fieldnames or []):
            raise ValueError("Input CSV needs image,mask,video_id; paths relative to --source")
        rows = list(reader)
    groups, stems = {}, set()
    for row in rows:
        if not row["video_id"].strip():
            raise ValueError("Every sample requires a video_id")
        for key in ("image", "mask"):
            path = Path(row[key])
            if path.is_absolute() or ".." in path.parts or not (source / path).is_file():
                raise ValueError(f"Invalid source path: {path}")
        stem = Path(row["image"]).stem
        if stem in stems:
            raise ValueError(f"Duplicate source image stem: {stem}; rename before splitting")
        stems.add(stem)
        with Image.open(source / row["image"]) as image:
            shape = (image.height, image.width)
        if read_mask(source / row["mask"]).shape != shape:
            raise ValueError(f"Mismatched dimensions: {row['image']}")
        groups.setdefault(row["video_id"].strip(), []).append(row)
    if len(groups) < 3:
        raise ValueError("At least three independent source videos are required")
    names = list(groups)
    random.Random(seed).shuffle(names)
    names.sort(key=lambda key: len(groups[key]), reverse=True)
    targets, counts = np.array([.8, .1, .1]) * len(rows), np.zeros(3, dtype=int)
    assignments = {}
    for i, video in enumerate(names):
        empty = np.flatnonzero(counts == 0)
        candidates = empty if len(names) - i == len(empty) else range(3)
        chosen = min(candidates, key=lambda k: sum((counts + np.eye(3, dtype=int)[k] * len(groups[video]) - targets) ** 2))
        assignments[video] = ("train", "val", "test")[chosen]
        counts[chosen] += len(groups[video])
    output_rows = []
    for row in rows:
        split = assignments[row["video_id"].strip()]
        stem = Path(row["image"]).stem
        destinations = {"image": Path(split) / "images" / (stem + Path(row["image"]).suffix),
                        "mask": Path(split) / "masks" / (stem + ".png")}
        for path in destinations.values():
            (output / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / row["image"], output / destinations["image"])
        Image.fromarray((read_mask(source / row["mask"]) * 255).astype(np.uint8)).save(output / destinations["mask"])
        output_rows.append({**{k: str(v) for k, v in destinations.items()}, "video_id": row["video_id"].strip(), "split": split})
    write_manifest(output / "manifest.csv", output_rows)
    report = {"seed": seed, "counts": dict(zip(("train", "val", "test"), counts.tolist())),
              "target_ratio": [0.8, 0.1, 0.1], "group_counts": {s: sum(x == s for x in assignments.values()) for s in ("train", "val", "test")},
              "video_assignments": assignments}
    (output / "split_report.json").write_text(json.dumps(report, indent=2))
    return report


def write_manifest(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "mask", "video_id", "split"])
        writer.writeheader()
        writer.writerows(rows)


def make_synthetic(output, count=4, size=64, seed=42):
    output = Path(output)
    if count < 1 or size < 32:
        raise ValueError("Synthetic count must be positive and size >= 32")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Synthetic output must be new or empty")
    rng, rows = np.random.default_rng(seed), []
    for split, n in (("train", count), ("val", max(1, count // 2)), ("test", max(1, count // 2))):
        for folder in ("images", "masks"):
            (output / split / folder).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            background = np.clip(rng.normal(35, 5, (size, size, 3)), 0, 255).astype(np.uint8)
            mask = Image.new("L", (size, size))
            draw = ImageDraw.Draw(mask)
            for _ in range(3):
                x, y = rng.integers(4, size - 12, size=2).tolist()
                if rng.random() < .5:
                    draw.ellipse((x, y, x+8, y+6), fill=255)
                else:
                    draw.line((x, y, x+8, y+10), fill=255, width=2)
            binary = np.asarray(mask) > 0
            background[binary] = [190, 215, 225]
            name = f"synthetic_{split}_{i:04d}.png"
            image_path, mask_path = f"{split}/images/{name}", f"{split}/masks/{name}"
            Image.fromarray(background).save(output / image_path)
            mask.save(output / mask_path)
            rows.append({"image": image_path, "mask": mask_path, "video_id": f"synthetic_{split}", "split": split})
    write_manifest(output / "manifest.csv", rows)
    (output / "SYNTHETIC_ONLY.json").write_text(json.dumps({"notice": "Synthetic plumbing test only; not research evidence", "seed": seed}, indent=2))
    return {"root": str(output), "images": len(rows), "synthetic": True}
