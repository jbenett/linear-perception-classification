import pickle

def flatten(nested_list):
    new_list = []
    for item in nested_list:
        if type(item) == type([]):
            new_list.extend(item)
        else:
            new_list.append(item)
    return new_list

attributemap = {
    'Weekday' : [1,0,0],
    'Saturday' : [0,1,0],
    'Sunday' : [0,0,1],
    'morning' : [1,0,0],
    'afternoon' : [0,1,0],
    'evening' : [0,0,1],
    '<30' : [1,0,0],
    '30-60' : [0,1,0],
    '>60' : [0,0,1],
    'silly' : [1,0,0],
    'happy' : [0,1,0],
    'tired' : [0,0,1],
    'no' : 0,
    'yes' : 1,
    'SettersOfCatan' : 1,
    'ApplesToApples' : 0
}

training_set = []
testing_set = []

print('reading training set')
with open('game_attrdata_train.dat') as training_file:
    for line in training_file:
        row = line.rstrip('\n').split(',')
        mapped_row = [attributemap[key] for key in row]
        classification = mapped_row.pop(len(mapped_row) - 1)
        vector = [flatten(mapped_row), classification]
        training_set.append(vector)

with open('train_data_dump', 'wb') as f:
    pickle.dump(training_set, f)

print('reading testing set')
with open('game_attrdata_test.dat') as testing_file:
    for line in testing_file:
        row = line.rstrip('\n').split(',')
        mapped_row = [attributemap[key] for key in row]
        classification = mapped_row.pop(len(mapped_row) - 1)
        vector = [flatten(mapped_row), classification]
        testing_set.append(vector)

with open('test_data_dump', 'wb') as f:
    pickle.dump(testing_set, f)
