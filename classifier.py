import pickle
import sys
from operator import add


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
    return correct / len(data_set) * 100.0


def train(data_set, epoch_n, learn_rate=1.0):
    t_weights = [0.0 for i in range(len(data_set[0][0]))]
    weights = t_weights.copy()
    for epoch in range(epoch_n):
        for vector in data_set:
            classification = classify(vector[0], weights)
            error = vector[1] - classification
            for j in range(len(weights)):
                weights[j] += learn_rate * error * vector[0][j]
            t_weights = list(map(add, weights, t_weights))
        w_acc = calc_accuracy(data_set, weights)
        print(w_acc)

    return weights, t_weights


num_epoch = int(sys.argv[1])

training_set = []
testing_set = []

with open('train_data_dump', 'rb') as f:
    training_set = pickle.load(f)

with open('test_data_dump', 'rb') as f:
    testing_set = pickle.load(f)

model, avg_model = train(training_set,  num_epoch)
print(model)
