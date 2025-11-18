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
git clone https://github.com/<your-user>/uq_reconstruction_demo.git
cd uq_reconstruction_sample
pip install -e .
