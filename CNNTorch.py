import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchinfo import summary
import torch.optim as optim

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

#Convolutional layers
class OneConv(nn.Module):
    def __init__(self):
        super(OneConv,self).__init__()
        self.conv = nn.Conv2d(in_channels=1,out_channels=9,kernel_size=(5,5))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(5184,10)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv(x))
        x = self.flatten(x)
        x = nn.functional.log_softmax(self.fc(x),dim=1)
        return x
net = OneConv()

summary(net,input_size=(1,1,28,28))

def train(model, train_loader, test_loader, epoch=5, device="cpu"):
    # Config the model to train
    model.to(device)
    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    
    for ep in range(epoch):
        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Rate
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / total
        
        # Saving data
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        
        print(f"Epoch {ep+1}/{epoch}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    return history

def plot_results(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["test_acc"], label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    
    plt.show()
    
history = train(net,train_loader,test_loader,epoch=5)
plot_results(history)

fig, ax = plt.subplots(1,9)
with torch.no_grad():
    p = next(net.conv.parameters())
    for i,x in enumerate(p):
        ax[i].imshow(x.detach().cpu()[0,...])
        ax[i].axis('off')
    plt.show()

#Multi-layered CNNs and pooling layers
class MultiLayerCNN(nn.Module):
    def __init__(self):
        super(MultiLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc = nn.Linear(320,10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 320)
        x = nn.functional.log_softmax(self.fc(x),dim=1)
        return x

net = MultiLayerCNN()
summary(net,input_size=(1,1,28,28))

history = train(net,train_loader, test_loader,epoch=5)