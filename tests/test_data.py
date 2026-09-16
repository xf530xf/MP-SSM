from copy import deepcopy
import csv

import numpy as np
from PIL import Image
import pytest

from mp_ssm.config import DEFAULTS
from mp_ssm.data import SegmentationDataset, audit_manifest, make_synthetic, read_mask, split_by_video, write_manifest


def test_masks_accept_both_encodings_and_reject_gray(tmp_path):
    path = tmp_path / "mask.png"
    for values in ([[0, 1]], [[0, 255]]):
        Image.fromarray(np.array(values, dtype=np.uint8)).save(path)
        np.testing.assert_array_equal(read_mask(path), [[0, 1]])
    Image.fromarray(np.array([[127]], dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="mask values"):
        read_mask(path)


def test_pairing_and_dimensions(tmp_path):
    make_synthetic(tmp_path, count=2)
    settings = {**DEFAULTS["data"], "image_size": 64}
    dataset = SegmentationDataset(tmp_path, "train", settings)
    assert dataset[0]["image"].shape == (3, 64, 64)
    assert dataset[0]["mask"].shape == (1, 64, 64)
    path = dataset.pairs[0][1]
    Image.new("L", (32, 32)).save(path)
    with pytest.raises(ValueError, match="dimensions"):
        SegmentationDataset(tmp_path, "train", settings)
    path.unlink()
    with pytest.raises(ValueError, match="missing masks"):
        SegmentationDataset(tmp_path, "train", settings)


def test_flip_is_synchronized_and_mixup_keeps_soft_labels(tmp_path):
    make_synthetic(tmp_path, count=2)
    settings = {**DEFAULTS["data"], "image_size": 64, "clahe": False, "hflip_p": 1,
                "vflip_p": 1, "rotate_p": 0, "mosaic_p": 0, "mixup_p": 0}
    plain = SegmentationDataset(tmp_path, "train", settings)
    augmented = SegmentationDataset(tmp_path, "train", settings, augment=True)
    first, flipped = plain[0], augmented[0]
    np.testing.assert_allclose(flipped["image"].numpy(), first["image"].numpy()[:, ::-1, ::-1])
    np.testing.assert_array_equal(flipped["mask"].numpy(), first["mask"].numpy()[:, ::-1, ::-1])
    augmented.settings.update({"mosaic_p": 1, "rotate_p": 1, "mixup_p": 1})
    labels = [augmented[0]["mask"].numpy() for _ in range(3)]
    assert any(np.any((mask > 0) & (mask < 1)) for mask in labels)
    assert all(np.isfinite(mask).all() and mask.min() >= 0 and mask.max() <= 1 for mask in labels)


def test_group_split_and_manifest_audit(tmp_path):
    source, output = tmp_path / "source", tmp_path / "split"
    source.mkdir()
    rows = []
    for i in range(12):
        image, mask = f"image_{i}.png", f"mask_{i}.png"
        Image.new("RGB", (32, 32), (i, 0, 0)).save(source / image)
        Image.new("L", (32, 32), 255).save(source / mask)
        rows.append({"image": image, "mask": mask, "video_id": f"video_{i // 2}"})
    manifest = source / "groups.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "mask", "video_id"])
        writer.writeheader()
        writer.writerows(rows)
    report = split_by_video(source, manifest, output)
    assert sum(report["counts"].values()) == 12
    assert min(report["counts"].values()) > 0
    assert len(report["video_assignments"]) == 6
    assert audit_manifest(output)["images"] == 12
    with (output / "manifest.csv").open() as stream:
        new_rows = list(csv.DictReader(stream))
    # Reassign provenance of one validation sample to a train video.
    train_video = next(row["video_id"] for row in new_rows if row["split"] == "train")
    next(row for row in new_rows if row["split"] == "val")["video_id"] = train_video
    write_manifest(output / "manifest.csv", new_rows)
    with pytest.raises(ValueError, match="Video leakage"):
        audit_manifest(output)


def test_refuse_split_without_video_ids(tmp_path):
    manifest = tmp_path / "bad.csv"
    manifest.write_text("image,mask\na.png,b.png\n")
    with pytest.raises(ValueError, match="video_id"):
        split_by_video(tmp_path, manifest, tmp_path / "out")
