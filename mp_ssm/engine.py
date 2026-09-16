"""Training and evaluation; test labels never influence checkpoint selection."""

from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import save_config, validate_config
from .data import SegmentationDataset, audit_manifest
from .metrics import RegionMetrics
from .model import MPSSM
from .runtime import (environment, load_model, read_checkpoint, resolve_device, restore_rng,
                      rng_state, save_checkpoint, seed_everything, write_json)


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


@torch.inference_mode()
def evaluate_loader(model, loader, device, settings):
    model.eval()
    metrics = RegionMetrics(**settings)
    loss_sum, pixel_count, per_image = 0.0, 0, []
    for batch in loader:
        images, targets = batch["image"].to(device), batch["mask"].to(device)
        logits = model(images)
        loss_sum += nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="sum").item()
        pixel_count += targets.numel()
        probabilities = logits.sigmoid().float().cpu().numpy()
        labels = targets.cpu().numpy()
        for name, probability, label in zip(batch["name"], probabilities, labels):
            per_image.append({"name": name, **metrics.update(probability, label)})
    return {**metrics.compute(), "bce": loss_sum / pixel_count, "per_image": per_image}


def validate_resume_config(old, new):
    # Backend/device may change for debugging; numerical trajectory is then not
    # guaranteed. Optimization, preprocessing and architecture must stay fixed.
    old, new = deepcopy(old), deepcopy(new)
    for config in (old, new):
        config["model"].pop("scan_backend", None)
        for key in ("epochs", "device", "cpu_threads", "workers", "log_interval", "save_interval"):
            config["train"].pop(key, None)
        config["data"].pop("root", None)
    if old != new:
        raise ValueError("Resume configuration differs in architecture, preprocessing, evaluation or optimization settings")


def train(config, output, resume=None):
    validate_config(config)
    t, data_settings = config["train"], config["data"]
    if data_settings["image_size"] <= 32 and t["batch_size"] == 1:
        raise ValueError("BatchNorm at the deepest stage requires training image_size >32 or batch_size >1")
    if t["amp"] and not t["device"].startswith("cuda"):
        raise ValueError("AMP is supported only on CUDA; paper defaults to amp=false")
    output = Path(output)
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError("Training output already contains files; use a new --output or explicitly --resume latest.pt")
    seed_everything(t["seed"], t["cpu_threads"])
    device = resolve_device(t["device"], config["model"]["scan_backend"])
    audit = audit_manifest(data_settings["root"])
    train_data = SegmentationDataset(data_settings["root"], "train", data_settings, augment=True)
    val_data = SegmentationDataset(data_settings["root"], "val", data_settings)
    generator = torch.Generator().manual_seed(t["seed"])
    loader = DataLoader(train_data, batch_size=t["batch_size"], shuffle=True,
                        num_workers=t["workers"], pin_memory=device.type == "cuda",
                        worker_init_fn=worker_seed, generator=generator)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=t["workers"], worker_init_fn=worker_seed)
    model = MPSSM(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], betas=tuple(t["betas"]),
                                 eps=t["eps"], weight_decay=t["weight_decay"], amsgrad=t["amsgrad"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["t_max"], eta_min=t["eta_min"])
    scaler = torch.amp.GradScaler("cuda", enabled=t["amp"])
    start_epoch, best_miou, global_step = 1, -1.0, 0
    if resume:
        checkpoint = read_checkpoint(resume)
        validate_resume_config(checkpoint["config"], config)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], generator)
        start_epoch, best_miou = checkpoint["epoch"] + 1, checkpoint["best_miou"]
        global_step = checkpoint["global_step"]
        if (output / "history.jsonl").exists():
            prior_rows = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines() if line.strip()]
            if prior_rows and max(row["epoch"] for row in prior_rows) >= start_epoch:
                raise ValueError("Output contains epochs newer than this checkpoint; resume into a new directory")
        # A resumed run in a new directory must retain its historical best,
        # even when subsequent epochs never exceed that validation score.
        prior_best_path = Path(resume).parent / "best.pt"
        if best_miou >= 0 and output.resolve() != Path(resume).resolve().parent:
            if not prior_best_path.exists():
                raise ValueError("Resuming into a new directory requires best.pt alongside the source checkpoint")
            prior_best = read_checkpoint(prior_best_path)
            validate_resume_config(prior_best["config"], config)
            if prior_best["epoch"] >= start_epoch or prior_best["best_miou"] != best_miou:
                raise ValueError("Source best.pt does not match this checkpoint's history; use the source latest.pt")
    if start_epoch > t["epochs"]:
        raise ValueError(f"Checkpoint already completed {start_epoch-1} epochs; increase --epochs")
    output.mkdir(parents=True, exist_ok=True)
    if resume and best_miou >= 0 and output.resolve() != Path(resume).resolve().parent:
        shutil.copy2(prior_best_path, output / "best.pt")
    save_config(config, output / "config.yaml")
    write_json(output / "environment.json", environment())
    write_json(output / "data_audit.json", audit)
    print(json.dumps({"device": str(device), "scan_backend": config["model"]["scan_backend"],
                      "parameters": sum(p.numel() for p in model.parameters()), "data_audit": audit}), flush=True)
    history = output / "history.jsonl"
    for epoch in range(start_epoch, t["epochs"] + 1):
        started = time.perf_counter()
        model.train()
        train_loss, seen = 0.0, 0
        epoch_lr = optimizer.param_groups[0]["lr"]
        for batch in loader:
            images, targets = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=t["amp"]):
                logits = model(images)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, step {global_step}")
            scaler.scale(loss).backward()
            if not t["amp"]:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"), error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.shape[0]
            seen += images.shape[0]
            global_step += 1
            if global_step % t["log_interval"] == 0:
                print(json.dumps({"epoch": epoch, "step": global_step, "train_bce": train_loss / seen, "lr": epoch_lr}), flush=True)
        scheduler.step()
        val, improved = None, False
        if epoch % t["val_interval"] == 0 or epoch == t["epochs"]:
            val = evaluate_loader(model, val_loader, device, config["eval"])
            write_json(output / f"validation_epoch_{epoch:04d}.json", val)
            improved = val["miou"] > best_miou
            if improved:
                best_miou = val["miou"]
        checkpoint = {"format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(generator),
                      "config": deepcopy(config), "epoch": epoch, "global_step": global_step,
                      "best_miou": best_miou, "data_audit": audit, "environment": environment()}
        save_checkpoint(checkpoint, output / "latest.pt")
        if improved:
            save_checkpoint(checkpoint, output / "best.pt")
        if epoch % t["save_interval"] == 0:
            save_checkpoint(checkpoint, output / f"epoch_{epoch:04d}.pt")
        row = {"epoch": epoch, "step": global_step, "train_bce": train_loss / seen,
               "lr": epoch_lr, "next_lr": scheduler.get_last_lr()[0], "best_miou": best_miou,
               "seconds": time.perf_counter() - started,
               "validation": {k: v for k, v in val.items() if k not in {"per_image", "thresholds", "f1_by_threshold"}} if val else None}
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
    return {"output": str(output), "completed_epochs": t["epochs"], "best_validation_miou": best_miou,
            "synthetic": audit["synthetic"]}


def evaluate(checkpoint_path, data_root, output, split="test", device_name=None, backend=None):
    model, config, device, checkpoint = load_model(checkpoint_path, device_name, backend)
    seed_everything(config["train"]["seed"], config["train"]["cpu_threads"])
    data_root = data_root or config["data"]["root"]
    audit = audit_manifest(data_root)
    dataset = SegmentationDataset(data_root, split, config["data"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    result = evaluate_loader(model, loader, device, config["eval"])
    result.update({"split": split, "checkpoint": str(Path(checkpoint_path).resolve()), "checkpoint_epoch": checkpoint["epoch"],
                   "data_audit": audit, "evaluation_size": config["data"]["image_size"],
                   "environment": environment(), "scan_backend": config["model"]["scan_backend"],
                   "note": "ODS/OIS optimize thresholds for reporting only; deployment threshold stays fixed"})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return {k: v for k, v in result.items() if k not in {"per_image", "thresholds", "f1_by_threshold"}}
