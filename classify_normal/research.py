import helpers_net
import classify_multi_late
import classify_multi_early
import classify_multi_v2

import torch.optim as optim
import torch.nn.functional as F

import os

EPOCH_COUNT = 8
ITER_COUNT = 10

multipliers = [1, 1.5, 0.5]
layers_counts = [0, 1, -1]
activ_funs = [F.relu, F.tanh]
opts = [optim.Adam, optim.SGD]
lrs = [0.001, 0.0001, 0.00001]
batch_sizes = [4, 32, 128]
params_list = [multipliers, layers_counts, activ_funs, opts, lrs, batch_sizes]
params_names = ['layers_pack', 'layers_count', 'activation_fun', 'optimizer', 'learning_rate', 'bach_size']

def std_dev(l):
    mean = sum(l) / len(l) 
    variance = sum([((x - mean) ** 2) for x in l]) / len(l) 
    std_dev = variance ** 0.5
    return std_dev

def run_research(filename, netModel, multiplier, layers_count, activ_fun, opt, lr, batch_size):
    print(f"Filename: {filename}")
    try:
        with open(filename, "a") as file:
            file.write('Accuracy, precision table, precision, f1, time\n')
            file.flush()
            results_acc = []
            results_prec = []
            results_f1 = []
            results_time = []
            for i in range(ITER_COUNT):
                print(f"iter: {i + 1}")
                trainloader, testloader = helpers_net.buildTrainTestLoader(batch_size)
                net = netModel(multiplier, layers_count, activ_fun)
                helpers_net.trainNet(net, opt, lr, trainloader, EPOCH_COUNT)
                accuracy, precision_table, f1, tim = helpers_net.getAccuracyPrecisionF1time(net, testloader)
                precision = sum(precision_table) / len(precision_table)
                results_acc.append(accuracy)
                results_prec.append(precision)
                results_f1.append(f1)
                results_time.append(tim)
                file.write(f'{accuracy}, {precision_table}, {precision}, {f1}, {tim}\n')
                file.flush()
            file.write('(avg, min, max, std deviation):\n')
            file.write(f"accuracy: {sum(results_acc) / ITER_COUNT}, {min(results_acc)}, {max(results_acc)}, {std_dev(results_acc)}\n")
            file.write(f"precision: {sum(results_prec) / ITER_COUNT}, {min(results_prec)}, {max(results_prec)}, {std_dev(results_prec)}\n")
            file.write(f"f1: {sum(results_f1) / ITER_COUNT}, {min(results_f1)}, {max(results_f1)}, {std_dev(results_f1)}\n")
            file.write(f"time: {sum(results_time) / ITER_COUNT}, {min(results_time)}, {max(results_time)}, {std_dev(results_time)}\n")
            file.flush()
    except Exception as error:
        print(f"Fail in: {filename}, error: {error}")


run_research("results/learning_rate42/early_fusion.log", classify_multi_early.Net, 1, 0, F.relu, optim.Adam, 0.00001, 4)
run_research("results/learning_rate42/mid_fusion.log", classify_multi_v2.Net, 1, 0, F.relu, optim.Adam, 0.00001, 4)
run_research("results/learning_rate42/late_fusion.log", classify_multi_late.Net, 1, 0, F.relu, optim.Adam, 0.00001, 4)

run_research("results/activation_fun21/early_fusion.log", classify_multi_early.Net, 1, 0, F.tanh, optim.Adam, 0.001, 4)

# run_research("results/late_fusion.log", classify_multi_late.Net, 1, 0, F.relu, optim.Adam, 0.001, 4)
# run_research("results/mid_fusion.log", classify_multi_v2.Net, 1, 0, F.relu, optim.Adam, 0.001, 4)
# run_research("results/early_fusion.log", classify_multi_early.Net, 1, 0, F.relu, optim.Adam, 0.001, 4)
# for i in range(len(params_list)):
#     params = [x[0] for x in params_list]
#     for j in range(1, len(params_list[i])):
#         params[i] = params_list[i][j]
#         dir_name = "results/" + params_names[i] + str(i) + str(j)
#         try:
#             os.mkdir(dir_name)
#         except:
#             pass

#         run_research(dir_name + "/late_fusion.log", classify_multi_late.Net, params[0], params[1], params[2], params[3], params[4], params[5])
#         run_research(dir_name + "/mid_fusion.log", classify_multi_v2.Net, params[0], params[1], params[2], params[3], params[4], params[5])
#         run_research(dir_name + "/early_fusion.log", classify_multi_early.Net, params[0], params[1], params[2], params[3], params[4], params[5])
