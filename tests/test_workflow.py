from copy import deepcopy
import json

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from mp_ssm.config import load_config
from mp_ssm.data import make_synthetic
from mp_ssm.engine import evaluate, train
from mp_ssm.inference import infer, regions_and_overlay
from mp_ssm.runtime import read_checkpoint


def test_train_resume_evaluate_image_and_video(tmp_path):
    config = load_config("configs/smoke.yaml")
    config["model"].update(channels=[2, 4, 6, 8], fusion_channels=4, d_state=2)
    data = tmp_path / "data"
    make_synthetic(data, count=2)
    config["data"]["root"] = str(data)
    # Two uninterrupted epochs must equal a one-epoch run resumed to epoch 2.
    full = deepcopy(config)
    full["train"]["epochs"] = 2
    train(full, tmp_path / "full")
    train(config, tmp_path / "resumed")
    train(full, tmp_path / "branched", tmp_path / "resumed/latest.pt")
    assert (tmp_path / "branched/best.pt").exists()
    train(full, tmp_path / "resumed", tmp_path / "resumed/latest.pt")
    uninterrupted = read_checkpoint(tmp_path / "full/latest.pt")
    resumed = read_checkpoint(tmp_path / "resumed/latest.pt")
    assert resumed["epoch"] == 2
    assert resumed["global_step"] == 4
    assert resumed["scheduler"] == uninterrupted["scheduler"]
    for name in uninterrupted["model"]:
        torch.testing.assert_close(resumed["model"][name], uninterrupted["model"][name], rtol=0, atol=0)
    branched = read_checkpoint(tmp_path / "branched/latest.pt")
    for name in uninterrupted["model"]:
        torch.testing.assert_close(branched["model"][name], uninterrupted["model"][name], rtol=0, atol=0)
    ckpt = tmp_path / "resumed/best.pt"
    result = evaluate(ckpt, str(data), tmp_path / "metrics.json")
    assert result["images"] == 1 and result["data_audit"]["synthetic"]
    assert 0 <= result["miou"] <= 1
    original = data / "test/images/synthetic_test_0000.png"
    odd = tmp_path / "odd.png"
    with Image.open(original) as image:
        image.resize((71, 53)).save(odd)
    infer(ckpt, odd, tmp_path / "image_output")
    probability = np.load(tmp_path / "image_output/odd_probability.npy")
    assert probability.shape == (53, 71)
    assert Image.open(tmp_path / "image_output/odd_mask.png").size == (71, 53)
    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 64))
    assert writer.isOpened()
    frame = cv2.imread(str(original))
    writer.write(frame)
    writer.write(frame)
    writer.release()
    report = infer(ckpt, video, tmp_path / "video_output")
    assert report["frames"] == 2
    lines = (tmp_path / "video_output/frames.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert "count_semantics" in json.loads((tmp_path / "video_output/metadata.json").read_text())
    capture = cv2.VideoCapture(str(tmp_path / "video_output/overlay.mp4"))
    assert capture.read()[0]
    capture.release()


def test_connected_regions_and_original_coordinates():
    probability = np.zeros((10, 12), dtype=np.float32)
    probability[2:5, 3:7] = .8
    probability[8, 10] = .9
    mask, overlay, objects = regions_and_overlay(np.zeros((10, 12, 3), dtype=np.uint8), probability)
    assert len(objects) == 2
    assert objects[0]["bbox_xyxy"] == [3, 2, 7, 5]
    assert objects[0]["area_pixels"] == 12
    assert objects[0]["mean_foreground_probability"] == pytest.approx(.8)
    assert mask.sum() == 13
    filtered, _, objects = regions_and_overlay(overlay, probability, min_area=2)
    assert filtered.sum() == 12 and len(objects) == 1
