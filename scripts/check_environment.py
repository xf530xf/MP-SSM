"""Run from any working directory after installing dependencies."""

import importlib.metadata
import json
import platform
import sys

import torch


report = {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
          "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
          "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
for package in ("mamba-ssm", "numpy", "opencv-python-headless", "PyYAML", "pytest"):
    try:
        report[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        report[package] = "not installed"
try:
    import selective_scan_cuda  # noqa: F401
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
    report["selective_scan_extension"] = "importable"
except (ImportError, RuntimeError, OSError) as exc:
    report["selective_scan_extension"] = f"unavailable: {exc}"
report["ready_for_cuda_training"] = torch.cuda.is_available() and report["selective_scan_extension"] == "importable"
print(json.dumps(report, indent=2))
