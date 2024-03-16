from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class MultimodalDataset(Dataset):
    def __init__(self, images_path, sensor_data, classes, transform=None):
        self.transform = transform
        self.images_path = images_path
        self.sensor_data = sensor_data
        self.classes = classes

    def __len__(self):
        return len(self.sensor_data)
    
    def __getitem__(self, idx):
        sensor_dat = np.array(self.sensor_data[idx][:3], dtype=np.float32)
        item_class = self.sensor_data[idx][3]
        image_name = self.sensor_data[idx][4] + '.png'
        image = Image.open(f'{self.images_path}/{item_class}/{image_name}')
        sample = (sensor_dat, image, self.classes.index(item_class))

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    
class Transform(object):
    def __call__(self, sample):
        sensor_data = sample[0]
        image = sample[1]
        label = sample[2]

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(size=(64, 64), antialias=True),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        return (torch.from_numpy(sensor_data), transform(image), label)

    
def buildTrainTestLoader(batch_size):
    classes = ['Mixture', 'NoGas', 'Perfume', 'Smoke']
    df = pd.read_csv('../dataset/sensor-data/Gas_Sensors_Measurements.csv')
    df = df.drop(['Serial Number', 'MQ5', 'MQ7', 'MQ8', 'MQ135'], axis=1)
    dataset = np.array(df)

    data_nogas = dataset[:1600]
    data_nogas_train = data_nogas[:1280]
    data_nogas_test = data_nogas[1280:]

    data_perfume = dataset[1600:3200]
    data_perfume_train = data_perfume[:1280]
    data_perfume_test = data_perfume[1280:]

    data_smoke = dataset[3200:4800]
    data_smoke_train = data_smoke[:1280]
    data_smoke_test = data_smoke[1280:]

    data_mixture = dataset[4800:]
    data_mixture_train = data_mixture[:1280]
    data_mixture_test = data_mixture[1280:]

    train_dataset = np.concatenate((data_nogas_train, data_perfume_train, data_smoke_train, data_mixture_train))
    np.random.shuffle(train_dataset)

    test_dataset = np.concatenate((data_nogas_test, data_perfume_test, data_smoke_test, data_mixture_test))
    np.random.shuffle(test_dataset)

    trainset = MultimodalDataset("../data/images", train_dataset, classes, transform=Transform())
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

    testset = MultimodalDataset("../data/images", test_dataset, classes, transform=Transform())
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=6)

    return trainloader, testloader

def getAccuracy(net, dataloader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            sensors_data, images, labels = data
            # calculate outputs by running images through the network
            outputs = net(sensors_data, images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

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
