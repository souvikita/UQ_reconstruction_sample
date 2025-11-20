# uq_reconstruction_demo/train.py

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .datasets import NoisyImageDataset
from .models import ConvAutoencoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out-dir", type=str, default="./checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:   
        device = torch.device("cpu")
    print(f"Using {device} device")

    transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
    dataset = NoisyImageDataset(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = ConvAutoencoder(dropout_p=0.1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for degraded, clean in loader:
            degraded = degraded.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            recon = model(degraded)
            loss = criterion(recon, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * degraded.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        print(f"Epoch {epoch}: loss={epoch_loss:.6f}")

        ckpt_path = out_dir / f"epoch_{epoch}.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    latest = out_dir / "latest.pt"
    torch.save({"model_state_dict": model.state_dict()}, latest)
    print(f"Saved latest checkpoint to {latest}")


if __name__ == "__main__":
    main()
