import helpers_net
import classify_multi_late
import classify_multi_early
import classify_multi_v2

import torch.optim as optim

EPOCH_COUNT = 8
ITER_COUNT = 10

def fill_base():
    batch_size = 4
    lr = 0.001
    opt = optim.Adam
    return batch_size, lr, opt

def run_research(filename, netModel, parametersFill):
    print(f"Filename: {filename}")
    try:
        batch_size, lr, opt = parametersFill()

        with open(filename, "a") as file:
            results = []
            for i in range(ITER_COUNT):
                print(f"iter: {i + 1}")
                trainloader, testloader = helpers_net.buildTrainTestLoader(batch_size)
                net = netModel()
                helpers_net.trainNet(net, opt, lr, trainloader, EPOCH_COUNT)
                accuracy = helpers_net.getAccuracy(net, testloader)
                results.append(accuracy)
                file.write(f'{accuracy}\n')
            file.write(f"avg: {sum(results) / ITER_COUNT}")
    except Exception as error:
        print(f"Fail in: {filename}, error: {error}")

run_research("results/late_fusion.log", classify_multi_late.Net, fill_base)
run_research("results/mid_fusion.log", classify_multi_v2.Net, fill_base)
run_research("results/early_fusion.log", classify_multi_early.Net, fill_base)
