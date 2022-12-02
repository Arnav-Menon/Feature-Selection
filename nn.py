import numpy as np

CLASS_LABELS = 0

best_subset = []
best_accuracy = 0

def traverseLatticeTree(data):
    current_set_of_features = []
    for i in range(1, len(data[0])):
        feature_to_add = 0
        best_so_far_accuracy = 0
        for j in range(1, len(data[0])):
            if j not in current_set_of_features:
                accuracy = leave_one_out_cv(data, current_set_of_features, [j])
                print(f"\tUsing feature(s) {current_set_of_features + [j]} accuracy is {accuracy}")

                if accuracy > best_so_far_accuracy:
                    global best_accuracy
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        global best_subset
                        best_subset = current_set_of_features + [j]
                    best_so_far_accuracy = accuracy
                    feature_to_add = j
        current_set_of_features.append(feature_to_add)
        print(f"\nFeature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy}\n")

    return [current_set_of_features, accuracy]

def leave_one_out_cv(data, current_set, feature_add):
    # the [0] is to get the class labels along with the features
    data = data[:, [0] + current_set + feature_add]
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

    # df = np.loadtxt("small_96.txt")
    # df = np.loadtxt("small_6.txt")
    df = np.loadtxt("small_88.txt")

    # df = np.loadtxt("large_21.txt")
    # df = np.loadtxt("large_96.txt")
    # df = np.loadtxt("large_96.txt")

    # df = np.loadtxt("small_data.txt")
    # df = np.loadtxt("large_data.txt")


    print("Beginning search.")
    subset, accuracy = traverseLatticeTree(df)
    print(f"Finished search! The best feature subset is {best_subset}, which has an accuracy of {best_accuracy}")
    # print(f"Finished search! The best feature subset is {subset}, which has an accuracy of {accuracy}")
