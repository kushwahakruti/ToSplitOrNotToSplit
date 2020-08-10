import numpy as np
import sys
import math
import csv

max_depth = None
attributes = None
label_values = None

class DecisionTree:
    def __init__(self, training_data, attributes, max_depth):
        self.training_data = training_data
        self.attributes = attributes
        self.max_depth = max_depth
        self.root = Node(training_data, 1, self.attributes, None, None)

    def train(self):
        self.root.train()

    def predict(self, test_data):
        predicted_labels = np.empty(len(test_data), dtype=object)
        i = 0
        for row in test_data:
            predicted_labels[i] = self.root.predict(row)
            i += 1
        return predicted_labels

    def plz_print(self):
        self.root.plz_print()
        return

class Node(DecisionTree):
    def __init__(self, training_data, depth, left_attributes, parent_attribute, parent_value):
        self.parent_training_data = training_data
        self.parent_attribute = parent_attribute
        self.parent_attribute_value = parent_value
        self.depth = depth
        self.attribute = None
        self.attribute_values = None
        self.left_attributes = left_attributes
        self.majority_vote = self.get_majority_vote(training_data)
        self.children = []

    def train(self):
        if self.depth > max_depth:
            return
        if len(self.left_attributes)==0:
            return
        entropy = calculate_entropy(self.parent_training_data, len(training_data[0])-1)
        if entropy == 0:
            return
        max_info = -1
        split_index = -1
        for i in self.left_attributes:
            conditional_entropy = calculate_conditional_entropy(self.parent_training_data, len(self.parent_training_data[0])-1, attributes.index(i))
            if entropy - conditional_entropy > max_info:
                split_index = attributes.index(i)
                max_info = entropy - conditional_entropy
        if max_info == 0:
            return
        self.attribute = attributes[split_index]
        self.attribute_values = set(self.parent_training_data[:, split_index])

        for value in self.attribute_values:
            split_training_data = self.parent_training_data[np.where(self.parent_training_data[:,split_index] == value)]

            cp = self.left_attributes[:]
            cp.remove(self.attribute)

            self.children.append(Node(split_training_data, self.depth + 1, cp, self.attribute, value))

        for child in self.children:
            child.train()
        return

    def predict(self, test_sample):
        if self.attribute == None:
            return self.majority_vote
        test_attribute_value = test_sample[attributes.index(self.attribute)]
        for index, value in enumerate(self.attribute_values):
            if test_attribute_value == value:
                return self.children[index].predict(test_sample)
        return self.majority_vote

    def get_majority_vote(self, training_data):
        dictionary = calculate_label_value_count(training_data, len(training_data[0])-1)
        max_value = max(dictionary.values())
        majority_list = []
        for i in dictionary:
            if dictionary[i] == max_value:
                majority_list.append(i)
        return max(majority_list)

    def plz_print(self):
        std_out = ""
        temp_dictionary = calculate_label_value_count(self.parent_training_data, len(self.parent_training_data[0])-1)
        if self.parent_attribute != None:
            std_out = "| "*(self.depth-1)+self.parent_attribute + " = " +self.parent_attribute_value + ": "
        std_out += "["
        for i in label_values:
            if i in temp_dictionary:
                std_out += (str)(temp_dictionary[i])+" "+i+"/"
            else:
                std_out += '0'+" "+i+"/"
        std_out = std_out[:-1]
        std_out += "]"+'\r'
        print(std_out)
        for child in self.children:
            child.plz_print()
        return

def calculate_entropy(data, index):
    label_count = calculate_label_value_count(data, index)
    entropy = 0
    for i in label_count:
        probability = label_count[i]/(float)(sum(label_count.values()))
        if probability != 0:
            entropy += -1*(probability*math.log(probability, 2))
    return entropy

def calculate_conditional_entropy(data, index, given_index):
    entropy = 0
    given_index_vote_dictionary = calculate_vote_dictionary(data, given_index)
    for i in given_index_vote_dictionary:
        given_index_probability = sum(given_index_vote_dictionary[i].values())/(float)(len(data))
        part_entropy = 0
        for j in given_index_vote_dictionary[i]:
            probability = given_index_vote_dictionary[i][j]/(float)(sum(given_index_vote_dictionary[i].values()))
            if probability != 0:
                part_entropy += -1*(probability*math.log(probability, 2))
        part_entropy *= given_index_probability
        entropy += part_entropy
    return entropy

def calculate_mutual_information(data, index, given_index):
    return calculate_entropy(data, index) - calculate_conditional_entropy(data, index, given_index)

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

def calculate_error(predicted_labels, true_labels):
    # error value = (count of incorrect assigned labels)/(total data rows)
    count = 0
    for i in range(0, len(predicted_labels)):
        if predicted_labels[i]!=true_labels[i]:
            count += 1
    return count/float(len(predicted_labels))


def read_data(input):
    with open(input, "r") as file:
        reader = csv.reader(file, delimiter='\t')
        header = next(reader)
        return header, np.array([row for row in reader])

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

if __name__ == "__main__":
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = (int)(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    # read data into matrix
    attributes, training_data = read_data(train_input)
    decisiontree = DecisionTree(training_data, attributes[:-1], max_depth)
    decisiontree.train()

    temp, test_data = read_data(test_input)

    # label values
    label_values = set(training_data[:,-1])

    # prediction
    train_labels = decisiontree.predict(training_data[:,:-1])
    output(train_labels, train_out)

    test_labels = decisiontree.predict(test_data[:,:-1])
    output(test_labels, test_out)

    decisiontree.plz_print()

    # error rate calculation
    with open(metrics_out, "a") as metrics_output:
        metrics_output.write("error(train): "+str(calculate_error(train_labels, training_data[:,-1]))+'\n')
        metrics_output.write("error(test): "+str(calculate_error(test_labels, test_data[:,-1]))+'\n')
