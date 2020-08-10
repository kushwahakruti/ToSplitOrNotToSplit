import numpy as np
import sys
import csv

class Node:
    def __init__(self, attribute, attribute_values, majority_vote_dictionary):
        self.attribute = attribute
        self.attribute_values = attribute_values
        self.majority_vote_dictionary = majority_vote_dictionary


def read_data(input):
    with open(input, "r") as file:
        reader = csv.reader(file, delimiter='\t')
        header = next(reader)
        return np.array([row for row in reader])

def create_decision_stump(training_data, split_index, label_values):

    # values that the split index attribute can take
    attribute_values = set(training_data[:,split_index])

    # create vote dictionary
    vote_dictionary = {}
    for value in attribute_values:
        vote_label = {}
        for l_value in label_values:
            vote_label[l_value] = 0
            vote_dictionary[value] = vote_label

    # add count of each label corresponding to attribute label to vote dictionary
    for value in attribute_values:
        for row in training_data:
            if row[int(split_index)] == value:
                vote_dictionary[value][row[-1]] += 1

    # calculate majority label for each attribute value
    majority_vote_dictionary = {}
    for value in vote_dictionary:
        majority_vote_dictionary[value] = max(vote_dictionary[value], key=vote_dictionary[value].get)

    # create decision stump with the majority vote result
    return Node(split_index, attribute_values, majority_vote_dictionary)


def predict(stump, data):
    #for each input row, assign label corresponding to the majority vote of that attribute value
    labels = np.empty(len(data), dtype=object)
    i = 0
    for row in data:
        labels[i] = stump.majority_vote_dictionary[row[stump.attribute]]
        i += 1
    return labels


def output(data, output_file):
    with open(output_file, "a") as output:
        for row in data:
            output.write(row+'\n')


def calculate_error(predicted_labels, true_labels):
    # error value = (count of incorrect assigned labels)/(total data rows)
    count = 0
    for i in range(0, len(predicted_labels)):
        if predicted_labels[i]!=true_labels[i]:
            count += 1
    return count/float(len(predicted_labels))


if __name__ == "__main__":
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    split_index = sys.argv[3]
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    # read data into matrix
    training_data = read_data(train_input)
    test_data = read_data(test_input)

    # label values
    label_values = set(training_data[:,-1])

    # create decision stump
    stump = create_decision_stump(training_data, int(split_index), label_values)

    # prediction
    train_labels = predict(stump, training_data)
    output(train_labels, train_out)

    test_labels = predict(stump, test_data)
    output(test_labels, test_out)

    # error rate calculation
    with open(metrics_out, "a") as metrics_output:
        metrics_output.write("error(train): "+str(calculate_error(train_labels, training_data[:,-1]))+'\n')
        metrics_output.write("error(test): "+str(calculate_error(test_labels, test_data[:,-1]))+'\n')

