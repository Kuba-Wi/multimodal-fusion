from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pandas as pd

import random


class AudioDataDataset(Dataset):
    def __init__(self, data, classes, transform=None):
        self.transform = transform
        self.data = data
        self.classes = classes

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dat = np.array(self.data[idx][1:11], dtype=np.float32)
        sample = (dat, self.classes.index(self.data[idx][106]))
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    

class Transform(object):
    def __call__(self, sample):
        transform = transforms.ToTensor()
        return (torch.from_numpy(sample[0]), sample[1])


batch_size = 4
classes = ['beach', 'city', 'classroom', 'football-match', 'forest', 'jungle', 'restaurant', 'river', 'grocery-store']
classes = [s.upper() for s in classes]
print(classes)

dataset = np.array(pd.read_csv('dataset.csv', header=None))
dataset = dataset[1:]
np.random.shuffle(dataset)

train_dataset = dataset[:(int(dataset.shape[0] * 0.8))]
test_dataset = dataset[(int(dataset.shape[0] * 0.8)):]

trainset = AudioDataDataset(train_dataset, classes, transform=None)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

testset = AudioDataDataset(test_dataset, classes, transform=None)
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
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 9)


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

PATH = './audio.pth'
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
