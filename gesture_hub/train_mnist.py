import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os

class Ge2019CNN(nn.Module):
    """
    CNN Architecture based on:
    "Design Of High Accuracy Detector For Mnist Handwritten Digit Recognition Based On Convolutional Neural Network"
    (Dong-yuan Ge et al., 2019)
    """
    def __init__(self):
        super(Ge2019CNN, self).__init__()
        
        # C1: Convolutional layer 1 (32 feature maps, 5x5 kernel) -> 24x24
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # S1: Subsampling layer 1 (Max Pooling) -> 12x12
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # C2: Convolutional layer 2 (64 feature maps, 5x5 kernel) -> 8x8
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        # S2: Subsampling layer 2 (Max Pooling) -> 4x4
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        
        # F1: Full connection 1 (1024 -> 2048)
        self.fc1 = nn.Linear(64 * 4 * 4, 2048)
        self.relu3 = nn.ReLU()
        
        # F2: Full connection 2 (2048 -> 784)
        self.fc2 = nn.Linear(2048, 784)
        self.relu4 = nn.ReLU()
        
        # Output layer (784 -> 10)
        self.fc3 = nn.Linear(784, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    print("Training Ge et al. (2019) CNN Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    model = Ge2019CNN().to(device)
    
    # The paper mentions a learning rate of 0.0007
    optimizer = optim.Adam(model.parameters(), lr=0.0007)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    # 5 epochs for a deep architecture
    for epoch in range(5):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
    save_path = os.path.join(os.path.dirname(__file__), 'mnist_mlp.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}!")
