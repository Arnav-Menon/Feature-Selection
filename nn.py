import numpy as np
import time

# these 2 keep track of the best subset of features and accuracy for that subset at a global level
best_subset = []
best_accuracy = 0

def forwardSelection(data):
    current_set_of_features = []
    # get accuracy on empty set, not necessarily 50%
    accuracy = (leave_one_out_cv(data, current_set_of_features, []))*100
    print(f"Accuracy on {current_set_of_features} is {accuracy:.1f}%")
    # start going through the tree of possible subsets
    for i in range(1, len(data[0])):
        feature_to_add = 0
        best_so_far_accuracy = 0
        # loop through features
        for j in range(1, len(data[0])):
            # only check the feature if it isn't already in our subset
            if j not in current_set_of_features:
                accuracy = (leave_one_out_cv(data, current_set_of_features, [j]))*100
                # update global and local subsets and accuracies to keep track of new improvement
                if accuracy > best_so_far_accuracy:
                    global best_accuracy
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        global best_subset
                        best_subset = current_set_of_features + [j]
                    best_so_far_accuracy = accuracy
                    feature_to_add = j
        current_set_of_features.append(feature_to_add)
        print(f"Feature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy:.1f}%")

    return [current_set_of_features, accuracy]

def backwardElimination(data):
    current_set_of_features = [i for i in range(1, len(data[0]))]
    # get accuracy on set of all features
    accuracy = (leave_one_out_cv(data, current_set_of_features, []))*100
    print(f"Accuracy on {current_set_of_features} is {accuracy:.1f}%")
    # start going through the tree of possible subsets
    for i in range(1, len(data[0])):
        feature_to_remove = 0
        best_so_far_accuracy = 0
        # loop through features
        for j in range(1, len(data[0])):
            # only check the feature if it is in our subset, bc we don't want to possible choose a subset with a feature that we've already eliminated
            if j in current_set_of_features:
                accuracy = (leave_one_out_cv(data, current_set_of_features, j, True))*100
                # make copy of list and modify that
                test = current_set_of_features[:]
                test.remove(j)
                # update global and local subsets and accuracies to keep track of new improvement
                if accuracy > best_so_far_accuracy:
                    global best_accuracy
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        global best_subset
                        best_subset = current_set_of_features[:]
                        best_subset.remove(j)
                    best_so_far_accuracy = accuracy
                    feature_to_remove = j
        current_set_of_features.remove(feature_to_remove)
        print(f"Feature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy:.1f}%")

# be flag is backward eleminiation flag lol, False by default
def leave_one_out_cv(data, current_set, feature_add, be=False):
    # the [0] is to get the class labels along with the features
    if be:
        # copy current_set into test and in different memory
        test = current_set[:]
        test.remove(feature_add)
        data = data[:, [0] + test]
    else:
        data = data[:, [0] + current_set + feature_add]

    number_correctly_identified = 0
    # loop through all data
    for i in range(len(data)):
        object_to_classify = data[i][1:]
        class_label = data[i][0]

        nearest_neighbor_distance = float("inf")
        nearest_neighbor_location = float("inf")
        # loop through all data
        for j in range(len(data)):
            if j != i:
                distance = np.sqrt(np.sum(np.square((object_to_classify - data[j][1:]))))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = j
                    nearest_neighbor_label = data[nearest_neighbor_location][0]
        if class_label == nearest_neighbor_label:
            number_correctly_identified += 1

    accuracy = number_correctly_identified / len(data)
    return accuracy

if __name__ == "__main__":
    test_file = str(input("What text file would you like to test? "))

    algo = int(input("What algorithm would you like to run, 1 or 2? \n\t 1) Forward Selection \n\t 2) Backward Elimination\n"))
    start = time.time()

    df = np.loadtxt(test_file)

    print(f"This dataset has {len(df[0] )- 1} features, not including class attributes, with {len(df)} instances\n")

    if algo == 1:
        print("Beginning forward search.")
        forwardSelection(df)
        print(f"Finished forward search! The best feature subset is {best_subset}, which has an accuracy of {best_accuracy:.1f}%\n")
        print(f"It took {time.time() - start:.1f} seconds.")
    else:
        print("Beginning backward search.")
        backwardElimination(df)
        print(f"Finished backward search! The best feature subset is {best_subset}, which has an accuracy of {best_accuracy:.1f}%\n")
        print(f"It took {time.time() - start:.1f} seconds.")