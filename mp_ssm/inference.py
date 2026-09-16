import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch

from .data import IMAGE_EXTENSIONS, file_index, preprocess_rgb
from .runtime import environment, load_model, seed_everything, write_json


@torch.inference_mode()
def predict_rgb(model, rgb, device, data_settings, native=False):
    height, width = rgb.shape[:2]
    prepared = preprocess_rgb(rgb, data_settings)
    if not native:
        prepared = cv2.resize(prepared, (data_settings["image_size"],) * 2, interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(np.ascontiguousarray(prepared.transpose(2, 0, 1))).float().unsqueeze(0).to(device) / 255.0
    logits = model(tensor)
    if logits.shape[-2:] != (height, width):
        logits = torch.nn.functional.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
    return logits.sigmoid()[0, 0].float().cpu().numpy()


def regions_and_overlay(rgb, probabilities, threshold=.5, min_area=1):
    if not 0 <= threshold <= 1 or min_area < 1:
        raise ValueError("threshold must be in [0,1], min_area >= 1")
    binary = (probabilities >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    retained = keep[labels].astype(np.uint8)
    probability_sums = np.bincount(labels.ravel(), weights=probabilities.ravel(), minlength=count)
    objects = []
    for index in range(1, count):
        x, y, w, h, area = map(int, stats[index])
        if area < min_area:
            continue
        objects.append({"region_id": len(objects) + 1, "bbox_xyxy": [x, y, x+w, y+h],
                        "area_pixels": area, "mean_foreground_probability": float(probability_sums[index] / area)})
    overlay = rgb.copy()
    selected = retained.astype(bool)
    overlay[selected] = (.55 * overlay[selected] + .45 * np.array([25, 140, 255])).astype(np.uint8)
    for obj in objects:
        x0, y0, x1, y1 = obj["bbox_xyxy"]
        cv2.rectangle(overlay, (x0, y0), (x1-1, y1-1), (25, 140, 255), 1)
        cv2.putText(overlay, f"{obj['mean_foreground_probability']:.2f}", (x0, max(10, y0-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (25, 140, 255), 1)
    return retained, overlay, objects


def save_frame(output, name, probabilities, mask, overlay=None):
    # Float32 array is the unthresholded probability; PNG is 0/255 foreground.
    np.save(output / f"{name}_probability.npy", probabilities.astype(np.float32))
    Image.fromarray(mask * 255).save(output / f"{name}_mask.png")
    if overlay is not None:
        Image.fromarray(overlay).save(output / f"{name}_overlay.png")


def infer(checkpoint_path, source, output, device_name=None, backend=None, native=False,
          threshold=None, min_area=1, max_frames=None, save_every=1):
    source, output = Path(source), Path(output)
    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Inference output must be new or empty")
    if save_every < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("save_every and max_frames must be positive")
    model, config, device, checkpoint = load_model(checkpoint_path, device_name, backend)
    seed_everything(config["train"]["seed"], config["train"]["cpu_threads"])
    threshold = config["eval"]["threshold"] if threshold is None else threshold
    if not 0 <= threshold <= 1 or min_area < 1:
        raise ValueError("Invalid threshold or minimum area")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"checkpoint": str(Path(checkpoint_path).resolve()), "checkpoint_epoch": checkpoint["epoch"],
                "trained_on_synthetic": checkpoint["data_audit"]["synthetic"], "source": str(source.resolve()),
                "native_resolution": native, "model_input_size": "native" if native else config["data"]["image_size"],
                "threshold": threshold, "min_area": min_area, "save_every": save_every,
                "scan_backend": config["model"]["scan_backend"], "device": str(device), "environment": environment(),
                "score_semantics": "mean pixel foreground probability, not calibrated detection confidence",
                "count_semantics": "connected regions per frame; no tracking or unique-particle count",
                "boxes": "xyxy, upper endpoints exclusive; touching particles may merge"}
    write_json(output / "metadata.json", metadata)
    if source.is_dir() or source.suffix.lower() in IMAGE_EXTENSIONS:
        paths = list(file_index(source).values()) if source.is_dir() else [source]
        if not paths:
            raise ValueError("No supported images found")
        rows = []
        for path in paths:
            with Image.open(path) as image:
                rgb = np.array(image.convert("RGB"))
            started = time.perf_counter()
            probability = predict_rgb(model, rgb, device, config["data"], native)
            mask, overlay, objects = regions_and_overlay(rgb, probability, threshold, min_area)
            seconds = time.perf_counter() - started
            save_frame(output, path.stem, probability, mask, overlay)
            rows.append({"name": path.name, "regions": len(objects), "objects": objects,
                         "preprocess_model_postprocess_seconds": seconds})
        write_json(output / "regions.json", rows)
        return {"output": str(output), "images": len(rows), "trained_on_synthetic": metadata["trained_on_synthetic"]}
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    writer, frame_index, expected_size = None, 0, None
    try:
        with (output / "frames.jsonl").open("w", encoding="utf-8") as stream:
            while max_frames is None or frame_index < max_frames:
                ok, bgr = capture.read()
                if not ok:
                    break
                height, width = bgr.shape[:2]
                if writer is None:
                    expected_size = (height, width)
                    # Many mp4 codecs require even dimensions. Pad only the video
                    # overlay; raw masks, arrays and box coordinates stay native.
                    writer = cv2.VideoWriter(str(output / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (width + width % 2, height + height % 2))
                    if not writer.isOpened():
                        raise RuntimeError("OpenCV cannot create mp4v output; check installed video codecs")
                if (height, width) != expected_size:
                    raise ValueError("Video frames changed dimensions")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                started = time.perf_counter()
                probability = predict_rgb(model, rgb, device, config["data"], native)
                mask, overlay, objects = regions_and_overlay(rgb, probability, threshold, min_area)
                seconds = time.perf_counter() - started
                encoded = cv2.copyMakeBorder(overlay, 0, height % 2, 0, width % 2, cv2.BORDER_REPLICATE)
                writer.write(cv2.cvtColor(encoded, cv2.COLOR_RGB2BGR))
                if frame_index % save_every == 0:
                    save_frame(output, f"frame_{frame_index:06d}", probability, mask)
                stream.write(json.dumps({"frame": frame_index, "nominal_time_seconds": frame_index / fps,
                                         "regions": len(objects), "objects": objects,
                                         "preprocess_model_postprocess_seconds": seconds}) + "\n")
                frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not frame_index:
        raise ValueError("Video contained no decodable frames")
    metadata.update({"processed_frames": frame_index, "source_fps": fps, "original_shape": list(expected_size)})
    write_json(output / "metadata.json", metadata)
    return {"output": str(output), "frames": frame_index, "trained_on_synthetic": metadata["trained_on_synthetic"]}
