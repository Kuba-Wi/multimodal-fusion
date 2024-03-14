import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def plot_accuracy_log():
    epochs = []
    accuracy_train = []
    accuracy_test = []

    with open("training.log", 'r') as file:
        for line in file:
            parts = line.strip().split(', ')
            epochs.append(int(parts[0]))
            accuracy_train.append(float(parts[1]) * 100)
            accuracy_test.append(float(parts[2]) * 100)

    # Tworzenie wykresu
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, accuracy_train, label='Zbiór treningowy', color='blue')
    plt.plot(epochs, accuracy_test, label='Zbiór testowy', color='red')

    plt.xlabel('Epoki', fontsize=14)
    plt.xlim(min(epochs), max(epochs) + 1)
    plt.xticks(range(min(epochs), max(epochs) + 1))

    plt.ylabel('Dokładność', fontsize=14)
    plt.ylim(0, 105)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    plt.yticks(range(0, 101, 10))

    plt.grid(True)

    max_point_train = (epochs[accuracy_train.index(max(accuracy_train))], round(max(accuracy_train), 2))
    max_point_test = (epochs[accuracy_test.index(max(accuracy_test))], round(max(accuracy_test), 2))

    plt.scatter(max_point_train[0], max_point_train[1], color='blue',
                label=f'Max ({max_point_train[0]}, {max_point_train[1]}%)')
    plt.scatter(max_point_test[0], max_point_test[1], color='red',
                label=f'Max ({max_point_test[0]}, {max_point_test[1]}%)')

    plt.title('Dokładność modelu w kolejnych epokach')
    plt.legend()
    plt.show()

plot_accuracy_log()
