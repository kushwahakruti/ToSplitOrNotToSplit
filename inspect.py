import numpy as np
import sys
import math
import csv

def read_data(input):
    with open(input, "r") as file:
        reader = csv.reader(file, delimiter='\t')
        header = next(reader)
        return header, np.array([row for row in reader])

def output(data, output_file):
    with open(output_file, "a") as output:
        for row in data:
            output.write(row+'\n')

def calculate_entropy(data, index):
    label_count = calculate_label_value_count(data, index)
    entropy = 0
    for i in label_count:
        probability = label_count[i]/(float)(sum(label_count.values()))
        if probability != 0:
            entropy += -1*(probability*math.log(probability, 2))
    return entropy

def calculate_label_value_count(training_data, index):
    # values that the split index attribute can take
    values = set(training_data[:,index])
    vote_label = {}
    for value in values:
        vote_label[value] = 0
    for row in training_data:
        vote_label[row[index]] += 1
    return vote_label

def calculate_vote_dictionary(training_data, split_index):
    # values that the split index attribute can take
    attribute_values = set(training_data[:,split_index])
    # create vote dictionary
    vote_dictionary = {}
    # add count of each label corresponding to attribute label to vote dictionary
    for value in attribute_values:
        vote_dictionary[value] = calculate_label_value_count(training_data[np.where(training_data[:,split_index] == value)], len(training_data[0])-1)

    return vote_dictionary

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    attributes, data = read_data(input_file)

    with open(output_file, "a") as metrics_output:
        metrics_output.write("entropy: "+str(calculate_entropy(data, len(data[0])-1))+'\n')
        metrics_output.write("error: "+str(min(calculate_label_value_count(data, len(data[0])-1).values())/(float)(sum(calculate_label_value_count(data, len(data[0])-1).values())))+'\n')
