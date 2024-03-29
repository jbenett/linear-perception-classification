import pickle
import sys
from operator import add
from numpy import *
from statistics import mean
import matplotlib.pyplot as plt


def classify(vector, weights, ignored_indexes):
    activator_sum = 0.0
    for i in range(len(vector)):
        if i not in ignored_indexes:
            activator_sum += weights[i] * vector[i]
    return 1.0 if activator_sum >= 0.0 else 0.0


def calc_accuracy(data_set, weights, ignored_indexes):
    correct = 1.0
    for vector in data_set:
        guess = classify(vector[0], weights, ignored_indexes)
        if vector[1] == guess:
            correct += 1.0
    return round(correct / len(data_set) * 100.0, 5)


def train(data_set, epoch_n, ignored_indexes):
    t_weights = [0.0 for i in range(len(data_set[0][0]))]
    weights = t_weights.copy()
    history = []
    averaged_history = []
    for epoch in range(epoch_n):
        average = []
        for vector in data_set:
            classification = classify(vector[0], weights, ignored_indexes)
            error = vector[1] - classification
            for j in range(len(weights)):
                weights[j] += error * vector[0][j]
            t_weights = list(map(add, weights, t_weights))
        history.append(weights.copy())
        for i in range(len(t_weights)):
            average.append((1 / (epoch_n * len(data_set))) * t_weights[i])
        averaged_history.append(average.copy())
    return history, averaged_history


ignored = []

num_epoch = int(sys.argv[1])

j = 2
while j < len(sys.argv):
    ignored.append(int(sys.argv[j]))
    j += 1

m_train_acc = []
t_train_acc = []

m_test_acc = []
t_test_acc = []

with open('train_data_dump', 'rb') as f:
    training_set = pickle.load(f)

with open('test_data_dump', 'rb') as f:
    testing_set = pickle.load(f)

model_history, avg_model_history = train(training_set,  num_epoch, ignored)

for n in range(num_epoch):
    m_train_acc.append(calc_accuracy(training_set, model_history[n], ignored))
    t_train_acc.append(calc_accuracy(training_set, avg_model_history[n], ignored))
    m_test_acc.append(calc_accuracy(testing_set, model_history[n], ignored))
    t_test_acc.append(calc_accuracy(testing_set, avg_model_history[n], ignored))

print('Average Accuracy Recorded for each Model:')
print('Current Model - Train:  ', mean(m_train_acc))
print('Current Model - Test:  ', mean(t_train_acc))
print('Avg Model - Train:  ', mean(m_test_acc))
print('Avg Model - Test:  ', mean(t_test_acc))


to_plot = [m_train_acc, t_train_acc, m_test_acc, t_test_acc]
labels = ['current model - train', 'avg model - train', 'current model - test', 'avg model - test']

for i in range(0, len(to_plot)):
    plt.plot(range(0, num_epoch), to_plot[i], label=labels[i])

plt.xlabel('epoch')
plt.ylabel('Accuracy')

plt.title("Perceptron Training Accuracy")
plt.legend()

plt.show()
