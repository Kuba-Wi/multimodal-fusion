import matplotlib.pyplot as plt


FILENAME = 'results_sensors'
result = []
with open(FILENAME + '.txt') as f:
    try:
        for line in f:
            result.append(int(line))
    except:
        pass

result_noise = []
with open(FILENAME + '_noise.txt') as f:
    try:
        for line in f:
            result_noise.append(int(line))
    except:
        pass


plt.ylim(0, 100)
plt.plot(result, label='no noise')
plt.plot(result_noise, label='with noise')
plt.legend()
plt.show()
