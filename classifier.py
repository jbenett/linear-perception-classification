import pickle
import sys
from operator import add
from numpy import *
import matplotlib.pyplot as plt


def classify(vector, weights):
    activator_sum = 0.0
    for i in range(len(vector)-1):
        activator_sum += weights[i] * vector[i]
    return 1.0 if activator_sum >= 0.0 else 0.0


def calc_accuracy(data_set, weights):
    correct = 1.0
    for vector in data_set:
        guess = classify(vector[0], weights)
        if vector[1] == guess:
            correct += 1.0
    return round(correct / len(data_set) * 100.0, 5)


def train(data_set, epoch_n, learn_rate=1.0):
    t_weights = [0.0 for i in range(len(data_set[0][0]))]
    weights = t_weights.copy()
    history = []
    averaged_history = []
    for epoch in range(epoch_n):
        average = []
        for vector in data_set:
            classification = classify(vector[0], weights)
            error = vector[1] - classification
            for j in range(len(weights)):
                weights[j] += learn_rate * error * vector[0][j]
            t_weights = list(map(add, weights, t_weights))
        history.append(weights.copy())
        for i in range(len(t_weights)):
            average.append((1 / (epoch_n * len(data_set))) * t_weights[i])
        averaged_history.append(average.copy())
    return history, averaged_history


num_epoch = int(sys.argv[1])

m_train_acc = []
t_train_acc = []

m_test_acc = []
t_test_acc = []

with open('train_data_dump', 'rb') as f:
    training_set = pickle.load(f)

with open('test_data_dump', 'rb') as f:
    testing_set = pickle.load(f)

model_history, avg_model_history = train(training_set,  num_epoch)

for n in range(num_epoch):
    m_train_acc.append(calc_accuracy(training_set, model_history[n]))
    t_train_acc.append(calc_accuracy(training_set, avg_model_history[n]))
    m_test_acc.append(calc_accuracy(testing_set, model_history[n]))
    t_test_acc.append(calc_accuracy(testing_set, avg_model_history[n]))

to_plot = [m_train_acc, t_train_acc, m_test_acc, t_test_acc]

for i in range(0, len(to_plot)):
    plt.plot(range(0, num_epoch), to_plot[i])

plt.xlabel('epoch')
plt.ylabel('Accuracy')

plt.title("Perceptron Training Accuracy")
plt.legend()

plt.show()

