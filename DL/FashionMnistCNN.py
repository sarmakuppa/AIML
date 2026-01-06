from scipy import optimize
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Mnist_CNN(nn.Module):
    # def __init__(self):
    #     super(Mnist_CNN, self).__init__()
    #     self.conv1 = nn.Conv2d( 1, 32, kernel_size=3, padding=1 )
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d( 32, 64, kernel_size=3, padding=1 )
    #     self.fc1 = nn.Linear(64 * 7 * 7, 128)
    #     self.fc2 = nn.Linear(128, 10)
    
    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))  # Output: 32x14x14
    #     x = self.pool(F.relu(self.conv2(x)))  # Output: 64x7x7
    #     x = x.view(-1, 64 * 7 * 7)
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7* 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize( ( 0.5, ), (0.5,) )
    ]
)

train_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

print(train_dataset.classes)

test_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=True
)

model = Mnist_CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs=3
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}], Loss: {running_loss:.4f}")

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")

def imshow(img):
    img = img / 2 + 0.5   # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(test_loader)
images, labels = next(dataiter)


print('Ground Truth: ' + '  '.join(str(labels[j].item()) for j in range(5)))

imshow(torchvision.utils.make_grid(images[:5]))
