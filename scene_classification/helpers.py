from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time


class MultimodalDataset(Dataset):
    def __init__(self, images_path, audio_data, classes, transform=None):
        self.transform = transform
        self.images_path = images_path
        self.audio_data = audio_data
        self.classes = classes

    def __len__(self):
        return len(self.audio_data)
    
    def __getitem__(self, idx):
        sensor_dat = np.array(self.audio_data[idx][1:11], dtype=np.float32)
        item_class = self.audio_data[idx][106]
        image_name = self.audio_data[idx][0]
        image = Image.open(f'{image_name}')
        sample = (sensor_dat, image, self.classes.index(item_class))

        if self.transform:
            sample = self.transform(sample)
        
        return sample


class Transform(object):
    def __call__(self, sample):
        audio_data = sample[0]
        image = sample[1]
        label = sample[2]

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(size=(32, 32), antialias=True),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        return (torch.from_numpy(audio_data), transform(image), label)

    
def buildTrainTestLoader(batch_size):
    classes = ['beach', 'city', 'classroom', 'football-match', 'forest', 'jungle', 'restaurant', 'river', 'grocery-store']
    classes = [s.upper() for s in classes]
    
    dataset = np.array(pd.read_csv('dataset.csv', header=None))
    dataset = dataset[1:]

    train_dataset = []
    test_dataset = []
    for cls in classes:
        x = [v for v in dataset if v[-1] == cls]
        np.random.shuffle(x)
        for line in x[:(int(0.8 * len(x)))]:
            train_dataset.append(line)
        for line in x[(int(0.8 * len(x))):]:
            test_dataset.append(line)

    train_dataset = np.array(train_dataset)
    test_dataset = np.array(test_dataset)

    trainset = MultimodalDataset("images_orginal", train_dataset, classes, transform=Transform())
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

    testset = MultimodalDataset("images_orginal", test_dataset, classes, transform=Transform())
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=6)

    return trainloader, testloader


def getAccuracyPrecisionF1time(net, dataloader):
    tp_table = [0 for _ in range(9)]
    fp_table = [0 for _ in range(9)]
    fn_table = [0 for _ in range(9)]
    precision_table = []
    recall_table = []
    correct = 0
    total = 0
    total_time = 0
    with torch.no_grad():
        for data in dataloader:
            sensors_data, images, labels = data
            start = time.time()
            outputs = net(sensors_data, images)
            _, predicted = torch.max(outputs.data, 1)
            end = time.time()
            pred_time = end - start
            total += labels.size(0)
            total_time += pred_time
            correct += (predicted == labels).sum().item()
            for i in range(predicted.size(0)):
                if predicted[i] == labels[i]:
                    tp_table[labels[i]] += 1
                else:
                    fn_table[labels[i]] += 1
                    fp_table[predicted[i]] += 1

    for i in range(len(tp_table)):
        precision_table.append(tp_table[i] / (tp_table[i] + fp_table[i]))
        recall_table.append(tp_table[i] / (tp_table[i] + fn_table[i]))

    precision = sum(precision_table) / len(precision_table)
    recall = sum(recall_table) / len(recall_table)
    f1 = 2 * (precision * recall) / (precision + recall)

    return correct / total, precision_table, f1, total_time

def trainNet(net, optim, lr, trainloader, epochCount):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim(net.parameters(), lr=lr)

    for epoch in range(epochCount):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            sensor, image, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(sensor, image)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(f"Epoch: {epoch + 1}")
