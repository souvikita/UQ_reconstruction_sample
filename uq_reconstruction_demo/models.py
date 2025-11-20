# uq_reconstruction_demo/models.py

import torch
import matplotlib.pyplot as plt
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Tiny convolutional autoencoder with dropout.
    Dropout stays active at test time for MC dropout UQ.
    """

    def __init__(self, dropout_p: float = 0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def enable_dropout(model: nn.Module) -> None:
    """
    Enable dropout layers during inference for MC dropout.
    """
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

def disable_dropout(model: nn.Module) -> None:
    """
    Disable dropout layers (put them back to eval mode).
    """
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.eval()
