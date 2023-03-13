import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
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
            # print(np.array(Image.fromarray(img))[14:114,33:95].shape)
            # from matplotlib import pyplot as plt
            # plt.imshow(np.array(Image.fromarray(img))[14:114,33:95], interpolation='nearest')
            # plt.show()
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
        self.conv1 = nn.Conv2d(3, 16, (12, 1), padding=(2, 0))
        self.pool1 = nn.MaxPool2d((4, 1))
        self.conv2 = nn.Conv2d(16, 64, (8, 1), padding=(1, 0))
        self.pool2 = nn.MaxPool2d((2, 1))
        self.conv3 = nn.Conv2d(64, 128, (4, 1), padding=(1, 0))
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
            val_acc, val_loss = self.evaluate_model(val_loader, device, criterion)
            train_acc, _ = self.evaluate_model(train_loader, device, criterion)
            print('Epoch {}/{} | Training Loss: {:.4f}  | Validation Acc: {:.4f}'.format(
                epoch + 1, epochs, avg_loss, val_acc))

            epoch_list.append(epoch + 1)
            trainloss_list.append(avg_loss)
            valacc_list.append(val_acc)
            trainacc_list.append(train_acc)
            valloss_list.append(val_loss)

            torch.save(model, f'/home/u956278/openai/openai_happy/classifier_model_saved_original_data/{epoch}')

    def evaluate_model(model, val_loader, device, criterion):
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            loss = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss += criterion(outputs, labels)

            val_loss = loss / len(val_loader)
            accuracy = 100 * correct / total
            return accuracy, val_loss.item()


# dictionary to save accuracies and losses
train_val_results = {}

for randomState in list(np.random.randint(0, high=2 ** 32 - 1, size=20)):
    epoch_list = []
    trainloss_list = []
    valacc_list = []
    trainacc_list = []
    valloss_list = []

    print('creating model')
    model = EmotionClassifier()
    print(model)
    print('loading images')
    X = load_images(file_path_1, file_path_2)
    print('splitting images')
    X_train, X_test, y_train, y_test = train_test_split(X, load_labels(X), test_size=0.2, random_state=randomState,
                                                        shuffle=True)
    train_dataset = ImageDataset(np.rollaxis(X_train, 3, 1), y_train)
    val_dataset = ImageDataset(np.rollaxis(X_test, 3, 1), y_test)
    print('building dataloader')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('training')

    model.train_model(train_loader, val_loader, epochs=20, lr=0.0001, device=device)

    train_val_results[randomState] = {}
    train_val_results[randomState]["epoch_list"] = epoch_list
    train_val_results[randomState]["trainacc_list"] = trainacc_list
    train_val_results[randomState]["trainloss_list"] = trainloss_list
    train_val_results[randomState]["valacc_list"] = valacc_list
    train_val_results[randomState]["valloss_list"] = valloss_list

with open('classif_graphs_data.pickle', 'wb') as handle:
    pickle.dump(train_val_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print(epoch_list, trainacc_list, trainloss_list, valacc_list, valloss_list)
""" 
    fig = plt.figure(figsize=(10, 6))

    plt.plot(epoch_list, trainacc_list, label = "Training Accuracy (%)")
    plt.plot(epoch_list, trainloss_list, label = "Training Loss")

    plt.legend()
    fig.suptitle('Training Loss', fontsize=20)
    plt.xlabel('Training Epoch', fontsize=16)
    plt.ylabel('', fontsize=16)


    plt.savefig('train.png')


    fig = plt.figure(figsize=(10, 6))

    plt.plot(epoch_list, valacc_list, label = "Validation Accuracy (%)")
    plt.plot(epoch_list, valloss_list, label = "Validation Loss")

    plt.legend()
    fig.suptitle('Validation Loss', fontsize=20)
    plt.xlabel('Training Epoch', fontsize=16)
    plt.ylabel('', fontsize=16)

    plt.savefig('val.png') """
