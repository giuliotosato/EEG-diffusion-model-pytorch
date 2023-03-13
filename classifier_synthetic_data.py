import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Define the model
class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (12, 1))
        self.pool1 = nn.MaxPool2d((4, 1))
        self.conv2 = nn.Conv2d(16, 64, (8, 1))
        self.pool2 = nn.MaxPool2d((2, 1))
        self.conv3 = nn.Conv2d(64, 128, (4, 1))
        self.pool3 = nn.MaxPool2d((2, 1))
        self.fc0 = nn.LazyLinear(out_features=5000)
        self.fc1 = nn.Linear(5000, 2500)
        self.fc2 = nn.Linear(2500, 1000)
        self.fc3 = nn.Linear(1000, 2)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = self.pool1(x)
        x = F.gelu(self.conv2(x))
        x = self.pool2(x)
        x = F.gelu(self.conv3(x))
        x = self.pool3(x)
        x = x.view((x.size(0), -1))
        x = F.gelu(self.fc0(x))
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_model(self, train_loader, val_loader, epochs, lr, device):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if (i + 1) % 32 == 0:
                    print('Batch {}/{} | Loss: {:.4f}'.format(i + 1, len(train_loader), running_loss / 32))
            avg_loss = running_loss / len(train_loader)
            # val_loss, val_acc = self.evaluate_model(val_loader, device)
            val_acc = self.evaluate_model(val_loader, device)
            print('Epoch {}/{} | Training Loss: {:.4f}  | Validation Acc: {:.4f}'.format(
                epoch + 1, epochs, avg_loss, val_acc))
            torch.save(model, f'/home/u956278/openai/openai_happy/classifier_model_saved/{epoch}')

    def evaluate_model(model, val_loader, device):
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            return accuracy


print('creating model')
model = EmotionClassifier()
# Set the directories containing the images


model = torch.load('/home/u956278/openai/openai_happy/classifier_model_saved/9', map_location=torch.device('cpu'))

for i in range(0, 12):
    dir1 = f'/home/u956278/openai/openai_happy/saved_models_happy/images_{i}'
    dir2 = f'/home/u956278/openai/openai_happy/saved_models_sad/images_{i}'

    # Load the images from both directories into a single array
    valX = []

    for dir in [dir1, dir2]:
        for file in os.listdir(dir):
            # Load the image
            image = cv2.imread(os.path.join(dir, file))
            # Convert the image to a tensor
            valX.append(np.array(Image.fromarray(image)))
    valX = np.array(valX)

    # Concatenate the tensors and pass them through the model
    valX = np.rollaxis(valX, 3, 1)
    valX = torch.from_numpy(valX)
    valX = valX.contiguous()
    valX = valX.float()

    with torch.no_grad():
        model.eval()
        y_pred = model(valX)
        # print(y_pred)
    # Count the number of correctly predicted samples in each directory
    target_labels = []

    dir1_count = 0
    dir2_count = 0
    misslabeled = 0
    for s, tensor in enumerate(y_pred):
        if tensor[0] < tensor[1] and s <= 128:
            dir1_count += 1
        elif tensor[0] > tensor[1] and s > 128:
            dir2_count += 1
        else:
            misslabeled += 1
    '''for i, tensor in enumerate(valX):
        prediction = y_pred[i].argmax()
        if i < len(os.listdir(dir1)):
            dir1_count += int(prediction == target_labels[i])
        else:
            dir2_count += int(prediction == target_labels[i])'''

    print(f"Number of correctly predicted samples in epoch{i}0 in {dir1}: {dir1_count}")
    print(f"Number of correctly predicted samples in epoch {i}0 in {dir2}: {dir2_count}")
    print(f"Number of misslabeled samples in epoch {i}0 in {dir2}: {misslabeled}")
    print(f"accuracy in epoch {i}0 : {(dir2_count + dir1_count) / (128 * 2)}")
