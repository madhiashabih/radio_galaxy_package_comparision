import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import time
from util import get_memory_usage, print_metrics
import sys

# Define the model structure similar to JAX
class NeuralNetPyTorch(nn.Module):
    def __init__(self, num_classes=10):
        super(NeuralNetPyTorch, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.avg_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.avg_pool2d(x, 2)
        x = x.view(-1, 16 * 5 * 5)  # Flatten
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

args = sys.argv  # Get the command-line arguments
run = args[1]

# Define dataset and preprocessing
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

dataiter = iter(trainloader)
images, labels = next(dataiter)
print(labels)

# Define model, optimizer, and loss
model = NeuralNetPyTorch()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
start_time = time.time()
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 0:
            print(f"Epoch {epoch + 1}, Batch {i}, Loss: {running_loss / 100:.3f}")
            running_loss = 0.0

train_time = time.time() - start_time

# Evaluation on test set
all_preds = []
all_labels = []
start_time = time.time()
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())
inference_time = time.time() - start_time
memory_usage = get_memory_usage()

# Results
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
cm = confusion_matrix(all_labels, all_preds)

print_metrics(acc, f1, cm, train_time, inference_time, memory_usage, "out/pytorch/mnist/" ,f"out/pytorch/mnist/output_{run}.txt")

