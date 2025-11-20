# uq_reconstruction_demo/infer.py

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt

from .datasets import NoisyImageDataset
from .models import ConvAutoencoder, enable_dropout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="./uq_outputs")
    return parser.parse_args()


def main():
    args = parse_args()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
    dataset = NoisyImageDataset(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ConvAutoencoder(dropout_p=0.1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    enable_dropout(model)  # keep dropout active

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (degraded, clean) in enumerate(loader):
        degraded = degraded.to(device)
        clean = clean.to(device)

        preds = []
        with torch.no_grad():
            for _ in range(args.num_samples):
                preds.append(model(degraded))
        preds = torch.stack(preds, dim=0)  # (S, 1, H, W)

        mean_pred = preds.mean(dim=0)
        var_pred = preds.var(dim=0)

        # quick visualization of one sample
        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        for ax in axes:
            ax.axis("off")

        axes[0].set_title("Degraded")
        axes[0].imshow(degraded[0, 0].cpu(), cmap="gray")

        axes[1].set_title("Clean")
        axes[1].imshow(clean[0, 0].cpu(), cmap="gray")

        axes[2].set_title("Mean recon")
        axes[2].imshow(mean_pred[0, 0].cpu(), cmap="gray")

        axes[3].set_title("UQ (variance)")
        axes[3].imshow(var_pred[0, 0].cpu(), cmap="magma")

        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{idx:04d}.png")
        plt.close(fig)

    print(f"Saved UQ visualizations to {out_dir}")


if __name__ == "__main__":
    main()
