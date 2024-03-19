import numpy as np
import pandas as pd

classes = ['beach', 'city', 'classroom', 'football-match', 'forest', 'jungle', 'restaurant', 'river', 'grocery-store']
classes = [s.upper() for s in classes]

dataset = np.array(pd.read_csv('dataset.csv', header=None))
dataset = dataset[1:]

x_train = []
x_test = []
for cls in classes:
    x = [v for v in dataset if v[-1] == cls]
    for line in x[:(int(0.8 * len(x)))]:
        x_train.append(line)
    for line in x[(int(0.8 * len(x))):]:
        x_test.append(line)

x_train = np.array(x_train)
x_test = np.array(x_test)

