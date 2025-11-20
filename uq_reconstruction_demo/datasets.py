# uq_reconstruction_demo/datasets.py
# Good practice for dataset implementation in PyTorch to be separated from training code.

from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image


class NoisyImageDataset(Dataset):
    """
    Simple dataset:
    - expects grayscale images in data_dir
    - applies a degradation transform (noise/compression)
    - target is the clean image, input is the degraded one
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        degrade_fn: Optional[Callable] = None,
        noise_std: float = 0.1,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.paths = sorted(
            p for p in self.data_dir.glob("*.png")
        )  # adjust to your format
        self.transform = transform
        self.degrade_fn = degrade_fn or self._default_degrade
        self.noise_std = noise_std

    def _default_degrade(self, img: torch.Tensor) -> torch.Tensor:
        # img: Tensor in [0, 1], shape (1, H, W)
        noise = torch.randn_like(img) * self.noise_std
        degraded = img + noise
        return degraded.clamp(0.0, 1.0)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("L")  # grayscale
        if self.transform is not None:
            img = self.transform(img)  # e.g. ToTensor()
        clean = img
        degraded = self.degrade_fn(clean)
        return degraded, clean
