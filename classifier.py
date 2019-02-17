import pickle
import sys


def classify(vector, weights):
    activator_sum = 0.0
    for i in range(len(vector)-1):
        activator_sum += weights[i] * vector[i]
    return 1.0 if activator_sum >= 0.0 else 0.0


def train_model(data_set, epoch_n):
    weights = [0.0 for i in range(len(data_set[0][0]))]
    for epoch in range(epoch_n):
        for vector in data_set:
            classification = classify(vector[0], weights)
    return weights


num_epoch = sys.argv[1]

training_set = []
testing_set = []

with open('train_data_dump', 'rb') as f:
    training_set = pickle.load(f)

with open('test_data_dump', 'rb') as f:
    testing_set = pickle.load(f)

model_weights = [0.0 for i in range(len(testing_set[0][0]))]
avg_weights = model_weights.copy()
