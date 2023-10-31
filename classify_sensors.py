from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pandas as pd

import os
import time


class SensorDataDataset(Dataset):
    def __init__(self, data, classes, transform=None):
        self.transform = transform
        self.data = data
        self.classes = classes

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dat = np.array(self.data[idx][1:8], dtype=np.float32)
        sample = (dat, self.classes.index(self.data[idx][8]))
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    

class Transform(object):
    def __call__(self, sample):
        transform = transforms.ToTensor()
        return (torch.from_numpy(sample[0]), sample[1])


batch_size = 4
classes = ['Mixture', 'NoGas', 'Perfume', 'Smoke']

dataset = np.array(pd.read_csv('dataset/sensor-data/Gas_Sensors_Measurements.csv', header=None))
dataset = dataset[1:]
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

print(train_dataset.shape)
print(test_dataset.shape)


trainset = SensorDataDataset(train_dataset, classes, transform=None)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

testset = SensorDataDataset(test_dataset, classes, transform=None)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=6)


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


class Net(nn.Module):
    def __init__(self):
	# initialize the network
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 4)


    def forward(self, x):
	# the forward propagation algorithm
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(4):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

PATH = './gas_sensor_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = next(dataiter)

print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

net = Net()
net.load_state_dict(torch.load(PATH))
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


# from sklearn.neural_network import MLPClassifier
# import numpy as np
# import pandas as pd

# if __name__ == '__main__':
#     model = MLPClassifier()
#     train_classes = train_dataset[:, 8]
#     for i, x in enumerate(train_classes):
#         train_classes[i] = classes.index(x)

#     print(train_classes)
#     print(train_classes.shape)
#     train_classes = np.array(train_classes, dtype=int)
#     train_dataset = np.array(train_dataset[:, 1:8], dtype=np.float32)
#     test_dataset_in = np.array(test_dataset[:, 1:8], dtype=np.float32)
#     test_classes = test_dataset[:, 8]
#     for i, x in enumerate(test_classes):
#         test_classes[i] = classes.index(x)

#     model = model.fit(train_dataset, train_classes)

#     errors = 0
#     for i, sample in enumerate(test_dataset_in):
#         prediction = model.predict([sample])
#         if prediction != test_classes[i]:
#             errors += 1
    
#     print(errors)
#     print((len(test_dataset_in) - errors) / len(test_dataset_in))
