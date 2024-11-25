import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Load MNIST using pytorch
def load_mnist(batch_size=128):
    train_dataset = torchvision.datasets.MNIST(root="data", train=True, transform=ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(root="data", train=False, transform=ToTensor(), download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Function that apply the filter to a convolution
def apply_filter_to_dataset(data_loader, kernel, title, num_images=5):
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add dimentions to convolution
    conv = nn.Conv2d(1, 1, kernel_size=kernel.shape[-1], bias=False)
    conv.weight = nn.Parameter(kernel)
    
    images, _ = next(iter(data_loader))  # Get the image batch
    images = images[:num_images]  # Select image
    
    #Applying filter
    filtered_images = conv(images).detach().squeeze().numpy()
    
    # Plot
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)
    for i in range(num_images):
        axes[0, i].imshow(images[i].squeeze(), cmap="viridis")
        axes[0, i].axis("off")
        axes[1, i].imshow(filtered_images[i], cmap="viridis")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Filted", fontsize=12)
    plt.show()

# Load data
train_loader, test_loader = load_mnist(batch_size=128)

# Visualize filter
apply_filter_to_dataset(
    train_loader,
    kernel=torch.tensor([[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]]),
    title="Vertical edge filter"
)

apply_filter_to_dataset(
    train_loader,
    kernel=torch.tensor([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]]),
    title="Horizontal edge filter"
)
