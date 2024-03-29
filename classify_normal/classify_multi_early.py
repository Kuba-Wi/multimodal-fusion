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

class Net(nn.Module):
    def __init__(self, multiplier=1, layers_count=0, activ_fun=F.relu):
        super(Net, self).__init__()
        self.layers_count = layers_count
        self.activ_fun = activ_fun

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) 
        self.fc1 = nn.Linear(3 + 16 * 13 * 13, int(184 * multiplier))
        self.fc2 = nn.Linear(int(184 * multiplier), int(212 * multiplier))
        self.fc3 = nn.Linear(int(212 * multiplier), int(106 * multiplier))
        self.fc4 = nn.Linear(int(106 * multiplier), 4)
        if layers_count == 1:
            self.fc4 = nn.Linear(int(106 * multiplier), int(54 * multiplier))
            self.fc5 = nn.Linear(int(54 * multiplier), 4)
        elif layers_count == -1:
            self.fc3 = nn.Linear(int(212 * multiplier), 4)

    def forward(self, sensor_data, image):
        image = self.pool(self.activ_fun(self.conv1(image)))
        image = self.pool(self.activ_fun(self.conv2(image)))
        image = image.view(-1, 16 * 13 * 13)

        combined = torch.cat([sensor_data, image], dim=1)
        combined = self.activ_fun(self.fc1(combined))
        combined = self.activ_fun(self.fc2(combined))
        if self.layers_count == 0:
            combined = self.activ_fun(self.fc3(combined))
            out = self.fc4(combined)
        elif self.layers_count == 1:
            combined = self.activ_fun(self.fc3(combined))
            combined = self.activ_fun(self.fc4(combined))
            out = self.fc5(combined)
        elif self.layers_count == -1:
            out = self.fc3(combined)

        return out

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

if __name__ == '__main__':
    batch_size = 4
    classes = ['Mixture', 'NoGas', 'Perfume', 'Smoke']

    df = pd.read_csv('dataset/sensor-data/Gas_Sensors_Measurements.csv')
    df = df.drop(['Serial Number', 'MQ5', 'MQ7', 'MQ8', 'MQ135'], axis=1)
    dataset = np.array(df)
    # dataset = dataset[1:]
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


    trainset = MultimodalDataset("data/images", train_dataset, classes, transform=Transform())
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

    testset = MultimodalDataset("data/images", test_dataset, classes, transform=Transform())
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=6)

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    dataiter = iter(trainloader)
    sensors_data, images, labels = next(dataiter)

    # show images
    # imshow(torchvision.utils.make_grid(images))
    # print labels
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))



    results = []
    with open("results_early.txt", "w") as file:
        for i  in range(50):
            net = Net()

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.001)
            # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            for epoch in range(4):  # loop over the dataset multiple times

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
                    if i % 100 == 99:    # print every 100 mini-batches
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                        running_loss = 0.0

            print('Finished Training')

            PATH = './multimodal_net.pth'
            torch.save(net.state_dict(), PATH)

            dataiter = iter(testloader)
            sensors_data, images, labels = next(dataiter)

            # print images
            # imshow(torchvision.utils.make_grid(images))
            print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

            net = Net()
            net.load_state_dict(torch.load(PATH))
            outputs = net(sensors_data, images)

            _, predicted = torch.max(outputs, 1)

            print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                        for j in range(4)))

            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in testloader:
                    sensors_data, images, labels = data
                    # calculate outputs by running images through the network
                    outputs = net(sensors_data, images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f'Accuracy of the network on the {total} test data: {100 * correct // total} %')
            results.append(100 * correct // total)
            file.write(f"{results[-1]}\n")

        avg = sum(results) / len(results)
        file.write(f"Avg: {avg}\n")
        

    # # prepare to count predictions for each class
    # correct_pred = {classname: 0 for classname in classes}
    # total_pred = {classname: 0 for classname in classes}

    # # again no gradients needed
    # with torch.no_grad():
    #     for data in testloader:
    #         sensors_data, images, labels = data
    #         outputs = net(sensors_data, images)
    #         _, predictions = torch.max(outputs, 1)
    #         # collect the correct predictions for each class
    #         for label, prediction in zip(labels, predictions):
    #             if label == prediction:
    #                 correct_pred[classes[label]] += 1
    #             total_pred[classes[label]] += 1


    # # print accuracy for each class
    # for classname, correct_count in correct_pred.items():
    #     accuracy = 100 * float(correct_count) / total_pred[classname]
    #     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
