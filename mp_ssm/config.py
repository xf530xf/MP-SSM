from copy import deepcopy
from pathlib import Path

import yaml


DEFAULTS = {
    "model": {"channels": [16, 32, 48, 64], "fusion_channels": 32, "d_state": 16,
              "expand": 2, "scan_backend": "cuda"},
    "data": {"root": "data/MP-NTNU", "image_size": 512, "clahe": True,
             "clahe_clip": 2.0, "clahe_grid": 8, "hflip_p": 0.5, "vflip_p": 0.5,
             "rotate_p": 0.5, "rotate_degrees": 15, "mosaic_p": 0.5,
             "mixup_p": 0.2, "mixup_alpha": 0.5},
    "train": {"device": "cuda", "epochs": 200, "batch_size": 1, "workers": 0,
              "seed": 42, "amp": False, "lr": 1e-3, "betas": [0.9, 0.999],
              "eps": 1e-8, "weight_decay": 1e-2, "amsgrad": False, "t_max": 50,
              "eta_min": 1e-5, "log_interval": 20, "val_interval": 10,
              "save_interval": 100, "cpu_threads": 4},
    "eval": {"threshold": 0.5, "threshold_start": 0.01, "threshold_end": 0.99,
             "threshold_step": 0.01},
}


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    config = deepcopy(DEFAULTS)
    for section, values in raw.items():
        if section not in config or not isinstance(values, dict):
            raise ValueError(f"Unknown or invalid config section: {section}")
        for name, value in values.items():
            if name not in config[section]:
                raise ValueError(f"Unknown config setting: {section}.{name}")
            config[section][name] = value
    validate_config(config)
    return config


def validate_config(config):
    data, train, evaluation = config["data"], config["train"], config["eval"]
    if not isinstance(data["image_size"], int) or data["image_size"] < 32:
        raise ValueError("image_size must be an integer >= 32")
    for key in ("hflip_p", "vflip_p", "rotate_p", "mosaic_p", "mixup_p"):
        if not 0 <= data[key] <= 1:
            raise ValueError(f"{key} must lie in [0,1]")
    if data["mixup_alpha"] <= 0 or data["clahe_clip"] <= 0 or data["clahe_grid"] < 1:
        raise ValueError("Invalid MixUp/CLAHE parameters")
    for key in ("epochs", "batch_size", "log_interval", "val_interval", "save_interval", "t_max", "cpu_threads"):
        if not isinstance(train[key], int) or train[key] < 1:
            raise ValueError(f"train.{key} must be a positive integer")
    if train["workers"] < 0 or train["lr"] <= 0:
        raise ValueError("Invalid workers or learning rate")
    if not 0 <= evaluation["threshold"] <= 1:
        raise ValueError("Evaluation threshold must lie in [0,1]")
    if not (0 < evaluation["threshold_start"] <= evaluation["threshold_end"] < 1):
        raise ValueError("Invalid ODS/OIS threshold range")
    if evaluation["threshold_step"] <= 0:
        raise ValueError("Threshold step must be positive")


def save_config(config, path):
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
