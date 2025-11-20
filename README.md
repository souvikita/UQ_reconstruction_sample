# UQ reconstruction sample

Minimal PyTorch demo for **uncertainty-aware image reconstruction** under
noise/compression, with an eye toward scientific imaging.

The core idea:

- Start from **degraded images** (noise/compression/masking)
- Train a small **convolutional autoencoder** to reconstruct the clean image
- Turn on **dropout at inference time** (MC dropout) to obtain:
  - a **mean reconstruction**
  - a **per-pixel uncertainty estimate** (variance across stochastic passes)

This is intentionally small and self-contained – a sketch of how I think about
**reliability and uncertainty quantification** in scientific ML.

## Quickstart (skeleton)

```bash
git clone https://github.com/<your-user>/UQ_reconstruction_sample.git
cd uq_reconstruction_sample
pip install -e .
```
All runnable code is under:

```
uq_reconstruction_demo/
```
---

## Training

Place grayscale PNG images into a directory, for example:

```
sample_images/
    img_000.png
    img_001.png
    ...
```
Then train the autoencoder:

```bash
python -m uq_reconstruction_demo.train \
    --data-dir sample_images \
    --out-dir checkpoints \
    --epochs 30
```
This will save a model checkpoint to:

```
checkpoints/latest.pt
```

Training runs on CPU, MPS (Apple Silicon), or CUDA automatically.

## MC Dropout Inference & Uncertainty Visualization

To compute **mean reconstructions** and **pixel-wise uncertainty maps**:

```bash
python -m uq_reconstruction_demo.infer \
    --data-dir sample_images \
    --checkpoint checkpoints/latest.pt \
    --num-samples 30 \
    --out-dir uq_outputs
```

Each output figure contains:

1. Degraded input  
2. Clean target image  
3. Mean reconstruction  
4. Variance heatmap (MC Dropout uncertainty)

Files are saved to:

```
uq_outputs/sample_0000.png
uq_outputs/sample_0001.png
...
```

---

## Model Architecture

A compact convolutional autoencoder featuring:

- Dropout in the encoder (for epistemic uncertainty)
- No dropout in the decoder (for stable reconstructions)
- Two downsampling stages: `128 → 64 → 32`
- Two upsampling stages via `ConvTranspose2d`, each followed by smoothing `Conv2d`
- Final `Sigmoid()` output constrained to `[0, 1]`

## Idiosyncracies

- Checkerboard patterns in mean recons. may appear due to transposed convolutions — this is expected for this minimal AE and acceptable for a demonstration.
- MC Dropout variance maps highlight structurally complex regions (edges, stripes, boundaries, gradients).
- This is a good baseline for scientific imaging UQ, before scaling to advanced models such as UNet, diffusion models, deep ensembles, or VAEs.

---

**Thanks to ChatGPT 5.1 for helping to create this nice README.**

