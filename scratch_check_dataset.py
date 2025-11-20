from torchvision import transforms as T
from uq_reconstruction_demo.datasets import NoisyImageDataset
import matplotlib.pyplot as plt
ds = NoisyImageDataset(
    data_dir="/Users/souvikb/various_analysis/UQ_reconstruction_sample/sample_png_image",
    transform=T.ToTensor(),
    noise_std=0.2,
)  # Example usage with increased noise standard deviation

print("Dataset length:", len(ds))
degraded, clean = ds[0]
print("Degraded shape:", degraded.shape, "min/max:", degraded.min().item(), degraded.max().item())
print("Clean shape    :", clean.shape, "min/max:", clean.min().item(), clean.max().item())

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(degraded.squeeze(), cmap='gray')
axs[0].set_title("Degraded Image")
axs[1].imshow(clean.squeeze(), cmap='gray')
axs[1].set_title("Clean Image")
plt.show()
