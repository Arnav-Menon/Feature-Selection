import numpy as np
import time
import random

CLASS_LABELS = 0

best_subset = []
best_accuracy = 0

def forwardSelection(data):
    current_set_of_features = []
    for i in range(1, len(data[0])):
        feature_to_add = 0
        best_so_far_accuracy = 0
        for j in range(1, len(data[0])):
            if j not in current_set_of_features:
                # maybe change j input to int and cast to list in loo_cv function
                accuracy = (leave_one_out_cv(data, current_set_of_features, [j]))*100
                # print(f"\tUsing feature(s) {current_set_of_features + [j]} accuracy is {accuracy:.1f}%")

                if accuracy > best_so_far_accuracy:
                    global best_accuracy
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        global best_subset
                        best_subset = current_set_of_features + [j]
                    best_so_far_accuracy = accuracy
                    feature_to_add = j
        current_set_of_features.append(feature_to_add)
        print(f"\nFeature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy:.1f}%")

    return [current_set_of_features, accuracy]

def backwardElimination(data):
    current_set_of_features = [i for i in range(1, len(data[0]))]
    for i in range(1, len(data[0])):
        feature_to_remove = 0
        best_so_far_accuracy = 0
        for j in range(1, len(data[0])):
            if j in current_set_of_features:
                accuracy = (leave_one_out_cv(data, current_set_of_features, j, True))*100
                test = current_set_of_features[:]
                test.remove(j)
                print(f"\tUsing feature(s) {test} accuracy is {accuracy:.1f}%")

                if accuracy > best_so_far_accuracy:
                    global best_accuracy
                    if accuracy < best_accuracy:
                        best_accuracy = accuracy
                        global best_subset
                        best_subset = current_set_of_features.remove(j)
                    best_so_far_accuracy = accuracy
                    feature_to_remove = j
        # print(f"Removing {feature_to_remove}")
        current_set_of_features.remove(feature_to_remove)
        print(f"Feature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy:.1f}%")
        # print(f"Current feature set {current_set_of_features}\n")
        # print("-----------------------------")

    # return [current_set_of_features, accuracy]

def leave_one_out_cv(data, current_set, feature_add, be=False):
    # the [0] is to get the class labels along with the features
    if be:
        # print(f"CS {current_set}")
        # print(f"FA {feature_add}")
        # copy current_set into test and in different memory
        test = current_set[:]
        test.remove(feature_add)
        # print(f"test {test}")
        data = data[:, [0] + test]
    else:
        data = data[:, [0] + current_set + feature_add]

    # print(data)
    # return random.randint(0, 10)
    number_correctly_identified = 0
    for i in range(len(data)):
        object_to_classify = data[i][1:]
        class_label = data[i][0]

        nearest_neighbor_distance = float("inf")
        nearest_neighbor_location = float("inf")
        for j in range(len(data)):
            if j != i:
                distance = np.sqrt(np.sum(np.square((object_to_classify - data[j][1:]))))
                # diff = object_to_classify - data[j][1:]
                # distance = np.sqrt(np.sum(np.dot(diff, diff)))
                # distance = np.sqrt(np.sum((object_to_classify - data[j][1:])**2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = j
                    nearest_neighbor_label = data[nearest_neighbor_location][0]
        
        if class_label == nearest_neighbor_label:
            number_correctly_identified += 1

    accuracy = number_correctly_identified / len(data)
    return accuracy

if __name__ == "__main__":
    start = time.time()

    # df = np.loadtxt("eamonn_test_data.txt")
    # df = np.loadtxt("eamonn_slides_data.txt")

    # df = np.loadtxt("small_test_data.txt")

    # df = np.loadtxt("small_96.txt")
    # df = np.loadtxt("small_6.txt")
    # df = np.loadtxt("small_88.txt")

    # df = np.loadtxt("large_21.txt")
    # df = np.loadtxt("large_96.txt")
    # df = np.loadtxt("large_6.txt")

    # my datasets
    df = np.loadtxt("small_data.txt")
    # df = np.loadtxt("large_data.txt")


    print("Beginning forward search.")
    forwardSelection(df)
    # figure out accuracy decimal point stuff here
    print(f"Finished forward search! The best feature subset is {best_subset}, which has an accuracy of {best_accuracy}%")
    print(f"It took {time.time() - start:.1f} seconds.")

    # print("Beginning backward search.")
    # backwardElimination(df)
    # print(f"Finished backward search! The best feature subset is {best_subset}, which has an accuracy of {best_accuracy}%")
    # print(f"It took {time.time() - start:.1f} seconds.")