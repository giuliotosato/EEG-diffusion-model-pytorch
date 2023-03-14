import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

file_path_1 = ("/home/u956278/openai/openai_happy/neural_imgs_sad")
file_path_2 = ("/home/u956278/openai/openai_happy/neural_imgs_happy")


# Load images as numpy arrays
def load_images(file_path_1, file_path_2):
    images = []
    for filename in os.listdir(file_path_1):
        img = cv2.imread(os.path.join(file_path_1, filename))
        if img is not None:
            images.append(np.array(Image.fromarray(img))[14:114, 33:95])
    for filename in os.listdir(file_path_2):
        img = cv2.imread(os.path.join(file_path_2, filename))
        if img is not None:
            images.append(np.array(Image.fromarray(img))[14:114, 33:95])
    return np.array(images)


# Load labels for images
def load_labels(X):
    labels_sad = [0 for i in range(X.shape[0] // 2)]
    labels_happy = [1 for i in range(X.shape[0] // 2, X.shape[0])]
    labels = labels_sad + labels_happy
    return labels


# Encode labels
def encode_labels(labels):
    return labels


class ImageDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return X, y


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
print('loading images')
X = load_images(file_path_1, file_path_2)
print('dataset size', X.shape)
print('splitting images')
X_train, X_test, y_train, y_test = train_test_split(X, load_labels(X), test_size=0.2, random_state=8, shuffle=True)
train_dataset = ImageDataset(np.rollaxis(X_train, 3, 1), y_train)
val_dataset = ImageDataset(np.rollaxis(X_test, 3, 1), y_test)
print('building dataloader')
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print('training')
model.train_model(train_loader, val_loader, epochs=10, lr=0.0001, device=device)
