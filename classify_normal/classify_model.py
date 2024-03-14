import torch
import torch.nn as nn
import torch.optim as optim
from helpers_net import buildTrainTestLoader
from classify_sensors2 import Net

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

LOG_FILE = 'training_sensors.log'
trainloader, testloader = buildTrainTestLoader(batch_size=4)

with open(LOG_FILE, "a") as log_file:
    log_file.write('\n')
    log_file.flush()

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(30):  # loop over the dataset multiple times
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

        train_accuracy = getAccuracy(net, trainloader)
        test_accuracy = getAccuracy(net, testloader)
        
        log_file.write(f'{epoch}, {train_accuracy}, {test_accuracy}\n')
        log_file.flush()
        print(epoch, train_accuracy, test_accuracy)

    print('Finished Training')

    PATH = './multimodal_net.pth'
    torch.save(net.state_dict(), PATH)
