"""Measured model-only latency, storage and memory; explicitly partial FLOPs."""

import io
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .model import MPSSM
from .runtime import environment, load_model, resolve_device, seed_everything, write_json


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def storage_summary(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "parameter_tensor_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "buffer_tensor_bytes": sum(b.numel() * b.element_size() for b in model.buffers()),
            "serialized_state_dict_bytes": buffer.tell()}


def benchmark(config, output, sizes=(512, 2048), warmup=10, iterations=50, precision="fp32",
              checkpoint_path=None, allow_slow_reference=False, profile_only=False):
    if warmup < 0 or iterations < 1 or not sizes or min(sizes) < 1:
        raise ValueError("Invalid benchmark sizes or iterations")
    seed_everything(config["train"]["seed"], config["train"]["cpu_threads"])
    if checkpoint_path:
        model, config, device, _ = load_model(checkpoint_path, config["train"]["device"], config["model"]["scan_backend"])
    else:
        device = resolve_device(config["train"]["device"], config["model"]["scan_backend"])
        model = MPSSM(**config["model"]).to(device).eval()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Unsupported precision")
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("Reduced precision benchmarking is supported on CUDA only")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bfloat16")
    if config["model"]["scan_backend"] == "reference" and max(sizes) > 128 and not allow_slow_reference and not profile_only:
        raise ValueError("Large reference scans are intentionally blocked; use CUDA, --profile-only, or --allow-slow-reference")
    result = {**storage_summary(model), "environment": environment(), "device": str(device),
              "scan_backend": config["model"]["scan_backend"], "precision": precision,
              "precision_mode": "autocast; master weights and SSM state parameters remain FP32",
              "checkpoint": str(checkpoint_path) if checkpoint_path else None,
              "scope": "batch=1, resident tensor -> model logits; excludes file I/O, preprocessing, transfer and postprocessing",
              "memory_scope": "CUDA allocator current and peak allocated/reserved bytes; CPU/MPS memory not measured",
              "flops_scope": "partial only: Conv2d and Linear multiply-adds at 2 FLOPs/MAC; excludes selective recurrence, "
                             "einsum projections, scan indexing, normalization, activations, pooling and interpolation; not total model FLOPs",
              "measurements": []}
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    if not profile_only:
        for size in sizes:
            hooks, counter = [], [0]
            def count_ops(module, inputs, output):
                if isinstance(module, nn.Conv2d):
                    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
                    counter[0] += output.numel() * kernel_ops * 2
                elif isinstance(module, nn.Linear):
                    counter[0] += output.numel() * module.in_features * 2
            try:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                sample = torch.randn(1, 3, size, size, device=device)
                with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=precision != "fp32"):
                    for _ in range(warmup):
                        model(sample)
                    synchronize(device)
                    for module in model.modules():
                        if isinstance(module, (nn.Conv2d, nn.Linear)):
                            hooks.append(module.register_forward_hook(count_ops))
                    prediction = model(sample)
                    if not torch.isfinite(prediction).all():
                        raise FloatingPointError("Nonfinite benchmark output")
                    del prediction
                    for hook in hooks:
                        hook.remove()
                    hooks.clear()
                    synchronize(device)
                    if device.type == "cuda":
                        base_allocated = torch.cuda.memory_allocated(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    milliseconds = []
                    for _ in range(iterations):
                        synchronize(device)
                        started = time.perf_counter()
                        prediction = model(sample)
                        synchronize(device)
                        milliseconds.append((time.perf_counter() - started) * 1000)
                        del prediction
                mean_ms = float(np.mean(milliseconds))
                row = {"size": [size, size], "warmup": warmup, "iterations": iterations,
                       "latency_mean_ms": mean_ms, "latency_median_ms": float(np.median(milliseconds)),
                       "latency_p95_ms": float(np.percentile(milliseconds, 95)), "images_per_second": 1000 / mean_ms,
                       "partial_conv_linear_flops": counter[0], "memory": None}
                if device.type == "cuda":
                    row["memory"] = {"baseline_allocated_bytes": base_allocated,
                                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                     "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
                result["measurements"].append(row)
                del sample
            except torch.cuda.OutOfMemoryError:
                result["measurements"].append({"size": [size, size], "status": "CUDA out of memory"})
                if "sample" in locals():
                    del sample
                torch.cuda.empty_cache()
            finally:
                for hook in hooks:
                    hook.remove()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result
