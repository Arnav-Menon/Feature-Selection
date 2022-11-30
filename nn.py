import numpy as np
import random

COLUMN = 1
NUM_FEATURES = 0

current_set_of_features = []

'''
leave one out cross validation
inputs:
output:
'''
def looCV():
    return random.randint(1, 10)

def traverseLatticeTree(data):
    for i in range(1, len(data[0])):
        print(f"On level {i} of the search tree")
        feature_to_add = 0
        best_so_far_accuracy = 0
        for j in range(1, len(data[0])):
            if j not in  current_set_of_features:
                print(f"--Considering adding feature {j}")
                accuracy = looCV()

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add = j
        current_set_of_features.append(feature_to_add)
        print(f"On level {i}, feature {feature_to_add} added to current set")

def kFoldCV():
    
    pass

if __name__ == "__main__":

    df = np.loadtxt("test_data.txt")
    # print(df)

    NUM_FEATURES = len(df[0]) - 1

    traverseLatticeTree(df)

