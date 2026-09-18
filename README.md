# MP-SSM





# Dataset Download

The dataset is available for download via Google Drive:

**[Download the dataset](https://drive.google.com/file/d/1QHemrIS6EpbCRMwih6pYyqF73lrvdQAo/view?usp=drive_link)**

Open the link above and click the download button to save the dataset to your device.



# MP-SSM: Binary Segmentation of Microplastics

A PyTorch implementation of the main-text method in *Real-Time Detection of Microplastics in Aquatic Environments Enabled by an Ultra-Lightweight Vision Model*. The project includes six-direction selective state-space recurrence, a four-stage VSSB backbone, SPPF, a multi-scale decoder, and MFF, together with training, evaluation, image/video inference, and benchmarking tools.

**This is a paper-guided reimplementation with documented engineering choices, not the original experimental source code. The manuscript's reported 0.95M parameters, 96.13% mIoU, 65 FPS, and memory figures are not treated as measured results for this implementation. The example checkpoints included in this workspace were trained on synthetic samples solely to verify the workflow.


## 1. Environment

Recommended environment for full training: Linux, Python 3.11, an NVIDIA GPU, and a CUDA-enabled PyTorch build compatible with the installed GPU driver. Use the reference backend on macOS or CPU for small correctness tests.


### NVIDIA server setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# First install a compatible CUDA-enabled PyTorch build:
# https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
python -m pip install packaging ninja einops
MAMBA_KEEP_CUDA_BUILD=TRUE python -m pip install 'mamba-ssm>=2.2,<3' --no-build-isolation
python scripts/check_environment.py
python -m pytest -q -m cuda
```

As described in the [official Mamba documentation](https://github.com/state-spaces/mamba), source installations may require explicitly enabling the CUDA extension. This project must be able to import `selective_scan_cuda`. If no compatible prebuilt wheel is available, installation requires the CUDA Toolkit, `nvcc`, and a compatible compiler. After successful installation, record the environment with `python -m pip freeze > environment.lock.txt`. Server driver/compiler compatibility has not been validated on the local Mac.

The implementation calls `selective_scan_fn` directly rather than using a complete Mamba, Mamba2, or Mamba3 block. The pure PyTorch reference backend implements the same recurrence without requiring Mamba. It processes sequence steps sequentially and is intended for small correctness tests, not high-resolution training. A missing CUDA extension produces an explicit error; the CUDA configuration does not silently fall back to the reference implementation.

## 2. Dataset Preparation

Use the following directory layout:

```text
data/MP-NTNU/
  train/images/0001.png
  train/masks/0001.png
  val/images/1001.png
  val/masks/1001.png
  test/images/2001.png
  test/masks/2001.png
  manifest.csv                 # Recommended for checking source-video isolation
```

Images may be PNG, JPEG, TIFF, or BMP and are converted to RGB. Each mask must be single-channel and contain only `0/1` or `0/255`. Images and masks are paired by filename stem, and their original dimensions must match. Training and standard evaluation resize both to the configured input size, using bilinear interpolation for images and nearest-neighbor interpolation for masks. Samples are loaded on demand rather than loading the entire dataset into memory.

Standard evaluation computes metrics on **512 × 512** masks, following the input setting in Appendix A.4. The original 2080 × 2080 images, inference benchmarks at 2048 × 2048, and evaluation at 512 × 512 represent different protocols and must be reported separately.

### Splitting by source video

For an unsplit dataset, prepare a CSV with paths relative to `--source`:

```csv
image,mask,video_id
images/video_a_0001.png,masks/video_a_0001.png,video_a
images/video_a_0002.png,masks/video_a_0002.png,video_a
images/video_b_0001.png,masks/video_b_0001.png,video_b
images/video_c_0001.png,masks/video_c_0001.png,video_c
```

```bash
python -m mp_ssm split --source /path/to/raw_dataset \
  --manifest /path/to/videos.csv --output data/MP-NTNU --seed 42
```

At least three independent source videos are required. The splitter assigns complete videos to approximate an 8:1:1 image-count ratio while keeping all three subsets nonempty. Unequal video lengths may prevent an exact ratio. It copies the files and writes `manifest.csv` and `split_report.json`, leaving the source data unchanged.

For an existing split, `manifest.csv` uses the columns `image,mask,video_id,split`. An example row is `train/images/0001.png,train/masks/0001.png,video_a,train`. Before training, the loader checks manifest coverage and verifies that a video does not appear in multiple subsets. Existing splits can also be loaded without a manifest, but `data_audit.json` explicitly marks video isolation as **unverified**. Source metadata must be accurate: the software cannot infer source videos or identify repeated observations of the same particle from the images alone.

## 3. Quick Workflow Check

The following commands create synthetic samples and exercise training, evaluation, and inference. This workspace already contains generated artifacts under `data/synthetic`, `runs/smoke`, and `runs/verification`. Use new output directories when repeating the workflow, or reuse the existing samples and checkpoints.

```bash
python -m mp_ssm synthetic --output data/synthetic_new --count 4 --size 64
python -m mp_ssm train --config configs/smoke.yaml \
  --data data/synthetic_new --output runs/smoke_new
python -m mp_ssm evaluate --checkpoint runs/smoke_new/best.pt \
  --data data/synthetic_new --output runs/smoke_new/test_metrics.json
python -m mp_ssm infer --checkpoint runs/smoke_new/best.pt \
  --source data/synthetic_new/test/images --output runs/smoke_new/predictions
```

`smoke.yaml` uses a smaller network, 64 × 64 inputs, and the CPU reference scan. It is not the full paper configuration. Synthetic data, results, and checkpoints carry a `synthetic` flag and must not be used as evidence of the manuscript's accuracy claims.

## 4. Training and Resuming

```bash
python -m mp_ssm train --config configs/paper.yaml \
  --data /path/to/MP-NTNU --output runs/mp_ssm_01
python -m mp_ssm train --config configs/paper.yaml \
  --data /path/to/MP-NTNU --output runs/mp_ssm_01 \
  --resume runs/mp_ssm_01/latest.pt --epochs 200
```

Defaults: one GPU, 512 × 512 inputs, 200 epochs, batch size 1, zero data-loader workers, seed 42, AMP disabled, `BCEWithLogitsLoss`, and AdamW. See `configs/paper.yaml` for the complete settings. Training logs the loss every 20 steps, validates every 10 epochs and at the final epoch, and saves an additional snapshot every 100 epochs. It updates `latest.pt` after every epoch and selects `best.pt` using **validation mIoU averaged over foreground and background**.

Checkpoints include the model, optimizer, learning-rate scheduler, AMP scaler, Python/NumPy/PyTorch/CUDA random states, DataLoader generator state, epoch, and step count. `--epochs` specifies the total target number of epochs, not the number of additional epochs. Resuming requires matching architecture, preprocessing, optimization, and evaluation settings. Changing devices or worker counts may prevent bitwise reproducibility.

When resuming into a new output directory, keep the matching historical `best.pt` alongside the source checkpoint. The program copies it so the previous best model remains available even if subsequent epochs do not improve. An older snapshot cannot overwrite an existing run history containing later epochs.

The scheduler uses `CosineAnnealingLR(T_max=50)` as specified in the appendix. Over a 200-epoch run, the learning rate rises again after epoch 50; this is not a single monotonic decay spanning all 200 epochs. The implementation retains `T_max=50`. Use a separate experiment configuration if a different schedule is intended.

Run outputs include `config.yaml`, `environment.json`, `data_audit.json`, `history.jsonl`, validation metrics, and checkpoints. Test-set evaluation is a separate command and does not run automatically during training.

## 5. Evaluation

```bash
python -m mp_ssm evaluate --checkpoint runs/mp_ssm_01/best.pt \
  --data /path/to/MP-NTNU --split test --output runs/mp_ssm_01/test_metrics.json
```

At the fixed threshold of 0.5, TP, FP, FN, and TN are accumulated across the dataset before computing precision, recall, F1, foreground IoU, background IoU, and their average, mIoU. **Foreground IoU and two-class mIoU are reported separately.** ODS is the best pooled dataset F1 obtained with one shared threshold; OIS is the mean of the best per-image F1 scores. The default threshold grid runs from 0.01 to 0.99 in steps of 0.01, and the complete threshold curve is saved.

ODS and OIS are evaluation statistics. An optimal threshold found on the test set is not written back to the inference configuration. Empty ground truth and empty prediction yield F1 = 1. A class absent from both prediction and ground truth receives IoU = 1. Precision is 0 when no foreground is predicted but foreground exists in the target; undefined recall is set to 1 when the target contains no foreground. These conventions are recorded in the result JSON.

## 6. Image and Video Inference

```bash
python -m mp_ssm infer --checkpoint runs/mp_ssm_01/best.pt \
  --source /path/to/image.png --output runs/image_result
python -m mp_ssm infer --checkpoint runs/mp_ssm_01/best.pt \
  --source /path/to/video.mp4 --output runs/video_result --save-every 10
```

By default, each image is resized to the training input size. The predicted logits are interpolated back to the original dimensions before applying sigmoid. Use `--native` to process the original resolution directly; this changes computation and input distribution, so results should be reported separately. The threshold and minimum displayed region area can be adjusted with `--threshold 0.5` and `--min-area 1`.

- **Images:** `*_probability.npy` at the original resolution in float32, `*_mask.png` with values 0/255, `*_overlay.png`, and region metadata in JSON.
- **Videos:** `overlay.mp4`, per-frame `frames.jsonl`, and saved probability arrays and masks. `--save-every` controls only the saving interval; every frame is still processed. Its default is 1. High-resolution probability arrays can require substantial disk space.
- **Metadata:** preprocessing settings, threshold, input size, device, and whether the checkpoint was trained on synthetic data.

Bounding boxes are derived from 8-connected foreground regions and use original-image `xyxy` coordinates with exclusive upper bounds. Each score is the mean foreground probability within a region, not a calibrated detection confidence. Touching particles may form one connected region. There is no tracking, so per-frame counts cannot be summed to obtain a count of unique particles. For odd-sized video frames, only the encoded overlay is padded by one row or column as needed; masks and box coordinates retain the original dimensions.

## 7. Parameters, Latency, and Memory

```bash
# Benchmark on CUDA. Add --checkpoint to measure a trained model.
python -m mp_ssm benchmark --config configs/paper.yaml \
  --sizes 512 2048 --warmup 10 --iterations 50 --precision fp32 \
  --output runs/benchmark_cuda.json

# On macOS, count parameters and storage without a high-resolution forward pass.
python -m mp_ssm benchmark --config configs/paper.yaml \
  --device cpu --scan-backend reference --profile-only \
  --output runs/profile_cpu.json
```

Latency measurement uses synchronization, warmup iterations, and batch size 1 with the input already resident on the device. It measures model forward execution and excludes file I/O, preprocessing, host-to-device transfer, and postprocessing. Results include mean, median, and P95 latency, plus throughput. CUDA reports baseline allocator usage and peak allocated/reserved memory. CPU/MPS memory is marked as unmeasured rather than reported as zero.

The FLOPs counter covers only Conv2d and Linear operations, using 2 FLOPs per multiply-accumulate, and reports them as `partial_conv_linear_flops`. It excludes selective recurrence, einsum projections, scan permutations, normalization, activations, pooling, and interpolation. **This value must not be presented as total model FLOPs.** FP16/BF16 benchmarks use autocast while retaining FP32 master weights; these runs do not demonstrate a halving of stored model weights.

## 8. Python API and Project Structure

```python
import torch
from mp_ssm import MPSSM

model = MPSSM(scan_backend="reference").eval()
with torch.inference_mode():
    logits = model(torch.randn(1, 3, 64, 96))
    probabilities = logits.sigmoid()
assert logits.shape == (1, 1, 64, 96)
```

```text
mp_ssm/       Model, selective scan, data, metrics, training, inference, benchmarks, and CLI
configs/      Paper configuration and CPU synthetic smoke-test configuration
tests/        Recurrence, shape, gradient, data isolation, metric, and workflow tests
scripts/      Environment diagnostics
docs/         Paper-to-code mapping and validation records
third_party/  Reference-project license notices
```

Full experimental validation requires the real image/mask dataset, reliable source-video metadata, and an NVIDIA server. Private data downloading and a complete 200-epoch training run are not performed automatically. Accuracy and performance claims must be based on actual measurements.
