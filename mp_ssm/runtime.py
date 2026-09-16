import json
import os
from pathlib import Path
import platform
import random
import sys

import cv2
import numpy as np
import torch

from .model import MPSSM
from .scan import require_cuda_scan


def seed_everything(seed, cpu_threads=4):
    torch.set_num_threads(cpu_threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cv2.setNumThreads(0)


def resolve_device(name, backend):
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use a NVIDIA server or --device cpu --scan-backend reference for small tests")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
    elif device.type != "cpu":
        raise ValueError("Supported devices: cpu, cuda[:index], mps")
    if backend == "cuda":
        if device.type != "cuda":
            raise ValueError("CUDA scan requires --device cuda; CPU/MPS need --scan-backend reference")
        require_cuda_scan()
    return device


def environment():
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": str(torch.__version__), "numpy": np.__version__, "opencv": cv2.__version__,
            "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def rng_state(generator):
    np_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
            "torch": torch.get_rng_state(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(checkpoint, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Expected an MP-SSM format_version=1 checkpoint")
    return checkpoint


def load_model(path, device_name=None, backend=None):
    checkpoint = read_checkpoint(path)
    config = checkpoint["config"]
    if backend:
        config["model"]["scan_backend"] = backend
    if device_name:
        config["train"]["device"] = device_name
    device = resolve_device(config["train"]["device"], config["model"]["scan_backend"])
    model = MPSSM(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config, device, checkpoint


def write_json(path, content):
    Path(path).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
