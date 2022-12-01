import numpy as np
import random

CLASS_LABELS = 0

# current_set_of_features = []

'''
leave one out cross validation
inputs:
output:
'''
def looCV():
    return random.randint(1, 10)

def traverseLatticeTree(data):
    current_set_of_features = []
    for i in range(1, len(data[0])):
        print(f"On level {i} of the search tree")
        feature_to_add = 0
        best_so_far_accuracy = 0
        for j in range(1, len(data[0])):
            if j not in current_set_of_features:
                print(f"--Considering adding feature {j}")
                accuracy = leave_one_out_cv(data, current_set_of_features, [j])

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add = j
        current_set_of_features.append(feature_to_add)
        # current_set_of_features.sort()
        print(f"On level {i}, feature {feature_to_add} added to current set")
        print("\t\t\t\t\t", current_set_of_features)
        print("\t\t\t\t\t", accuracy)
        # print(df[:, current_set_of_features], "\n\n")

# def kFoldCV(data):
def leave_one_out_cv(data, current_set, feature_add):
    # the [0] is to get the class labels along with the features
    data = data[:, [0] + current_set + feature_add]
    number_correctly_identified = 0
    for i in range(len(data)):
        object_to_classify = data[i][1:]
        # print(object_to_classify)
        class_label = int(data[i][0])
        # print(f"Looping over i at location {i}")
        # print(f"\tThe {i}th object is in class {class_label}")

        nearest_neighbor_distance = float("inf")
        nearest_neighbor_location = float("inf")
        for j in range(len(data)):
            if j != i:
                # print(f"Ask if {i} is nearest neighbor with {j}")
                # print(data[j][1:])
                distance = np.sqrt(np.sum(np.square((object_to_classify - data[j][1:]))))
                # print("\t", distance)
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = j
                    nearest_neighbor_label = int(data[nearest_neighbor_location][0])
        
        if class_label == nearest_neighbor_label:
            number_correctly_identified += 1

        # print(f"Object {i} is class {class_label}")
        # print(f"\tIt's nearest neighbor is {nearest_neighbor_location} which is in class {nearest_neighbor_label}")
    accuracy = number_correctly_identified / len(data)
    # print(f"accuracy: {accuracy}")
    return accuracy

if __name__ == "__main__":

    df = np.loadtxt("large_96.txt")

    NUM_FEATURES = len(df[0]) - 1

    # print(type(df))
    # print(df.shape)
    # print(df[:, [1,5]])
    traverseLatticeTree(df)
    # print(f"Accuracy is {kFoldCV(df)}")

