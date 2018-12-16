import numpy as np
from random import randint, choice
try:
    import pydot
    pydot_installed = True
except:
    pydot_installed = False
    print('To see visualisations install pydot (see report).\n')

import pickle

#######################################              Decision Tree              #######################################
class Tree:
    def __init__(self, op):
        self.op = op  # label of the attribute that is being tested
        self.kids = [0, 0]  # left and right children of the current node
        self.label = None  # label for leaf nodes
        self.prob = None

    def getLeftChild(self):
        return self.kids[0]

    def getRightChild(self):
        return self.kids[1]

    def getOp(self):
        return self.op

    def getLabel(self):
        return self.label

    def setOp(self, value):
        self.op = value

    def setLabel(self, value):
        self.label = value


def subset(x_data, y_data, target):
    # x_data(array[N,A]): the data for which a subset should be created with
    #                     N examples and A attributes
    # y_data(array[N]):   the labeled data for all N examples
    # target(range(1,7)): the target emotion

    bool_array = y_data == target
    binary_target = np.zeros(x_data.shape[0])
    binary_target[bool_array[:, 0]] = 1
    return binary_target


def entropy(p, n):
    if (p + n) == 0:
        return 0
    elif p == 0:
        return -n / (p + n) * np.log2(n / (p + n))
    elif n == 0:
        return -p / (p + n) * np.log2(p / (p + n))
    else:
        return -p / (p + n) * np.log2(p / (p + n)) - n / (p + n) * np.log2(n / (p + n))


def majority_value(binary_target):
    return int(round(np.sum(binary_target) / len(binary_target)))


def choose_best_decision_attribute(examples, attributes, binary_target):
    max_gain = -1
    p = np.sum(binary_target)
    n = len(binary_target) - p
    I = entropy(p, n)
    max_index = -1
    for i in range(len(attributes)):
        # the number of positive examples for the subset of the training data
        # for which the attribute has the value 0
        p0 = np.sum(binary_target[examples[:, i] == 0] == 1)
        # the number of negative examples for the subset of the training data
        # for which the attribute has the value 0
        n0 = np.sum(binary_target[examples[:, i] == 0] == 0)
        # the number of positive examples for the subset of the training data
        # for which the attribute has the value 1
        p1 = np.sum(binary_target[examples[:, i] == 1] == 1)
        # the number of negative examples for the subset of the training data
        # for which the attribute has the value 1
        n1 = np.sum(binary_target[examples[:, i] == 1] == 0)
        current_remainder = (p0 + n0) / (p + n) * entropy(p0, n0) + (p1 + n1) / (p + n) * entropy(p1, n1)
        current_gain = I - current_remainder
        if current_gain > max_gain:
            max_gain = current_gain
            max_index = i
    return max_index


def decision_tree_learning(examples, attributes, binary_target):
    # examples (array[N,A])     : examples with N number of examples,
    #                             A number of attributes (Action Units)
    # attributes (array[1,A])   : a vector with all available attributes A
    # binary_target (array[N,1]):

    if np.all(binary_target == binary_target[0], axis=0):
        tree = Tree('')
        tree.setLabel(int(binary_target[0]))
        tree.prob = np.mean(binary_target)
        return tree

    elif len(attributes) == 0:
        mode = majority_value(binary_target)

        tree = Tree('')
        tree.setLabel(mode)
        tree.prob = np.mean(binary_target)

        return tree


    else:
        best_attribute = choose_best_decision_attribute(examples, attributes, binary_target)
        op = attributes[best_attribute]
        tree = Tree(op)
        tree.prob = np.mean(binary_target)

        examples_0 = examples[examples[:, best_attribute] == 0]

        binary_target0 = binary_target[examples[:, best_attribute] == 0]

        examples_1 = examples[examples[:, best_attribute] == 1]

        binary_target1 = binary_target[examples[:, best_attribute] == 1]

        if (len(examples_0) == 0) or (len(examples_1) == 0):

            mode = majority_value(binary_target)

            tree = Tree('')
            tree.setLabel(mode)
            tree.prob = np.mean(binary_target)

        else:
            leftTree = decision_tree_learning(examples_0, attributes, binary_target0)
            tree.kids[0] = leftTree

            rightTree = decision_tree_learning(examples_1, attributes, binary_target1)
            tree.kids[1] = rightTree

    return tree


#######################################         Prediction         #######################################
def prediction(decision_tree, x_data):
    while decision_tree.getLabel() == None:
        op = decision_tree.getOp()
        if (x_data[op] == 0):
            decision_tree = decision_tree.getLeftChild()
        else:
            decision_tree = decision_tree.getRightChild()
    return decision_tree.getLabel()

def probability(decision_tree, x_data):
    prob = decision_tree.prob
    while decision_tree.getLabel() == None:
        op = decision_tree.getOp()
        prob = (prob + decision_tree.prob)/2
        if (x_data[op] == 0):
            decision_tree = decision_tree.getLeftChild()
        else:
            decision_tree = decision_tree.getRightChild()
    return prob

def test_accuracy(x_test, y_test, decision_tree):
    predictions = []
    for i in range(x_test.shape[0]):
        predictions.append(prediction(decision_tree, x_test[i]))
    predictions = np.array(predictions)
    return np.sum(predictions == y_test) / len(y_test)


def predict_point(T, x_data):
    # Choose first emotion found
    for i in range(6):
        if prediction(T[i], x_data) == 1:
            return i + 1
    return randint(1, 6)


def predict_point2(T, x_data):
    # Choose randomly from predicted emotions
    predicted_emotions = []
    for i in range(6):
        if prediction(T[i], x_data) == 1:
            predicted_emotions.append(i + 1)
    if len(predicted_emotions) == 0:
        return randint(1, 6)
    else:
        return choice(predicted_emotions)

def predict_point3(T, x_data):
    # Choose most likely from predicted emotions based on
    # proportion of positives at previous nodes
    predicted_emotions = []
    for i in range(6):
        predicted_emotions.append((probability(T[i], x_data), i + 1))
    if len(predicted_emotions) == 0:
        return randint(1, 6)
    else:
        return max(predicted_emotions)[1]

def testTrees(T, x2):
    # return predictions for all data points using predict_point
    predictions = []
    for j in range(x2.shape[0]):
        predictions.append(predict_point(T, x2[j]))
    predictions = np.reshape(np.array(predictions), [x2.shape[0], 1])
    return predictions

def testTrees2(T, x2):
    # return predictions for all data points using predict_point2
    predictions = []
    for j in range(x2.shape[0]):
        predictions.append(predict_point2(T, x2[j]))
    predictions = np.reshape(np.array(predictions), [x2.shape[0], 1])
    return predictions

def testTrees3(T, x2):
    # return predictions for all data points using predict_point3
    predictions = []
    for j in range(x2.shape[0]):
        predictions.append(predict_point3(T, x2[j]))
    predictions = np.reshape(np.array(predictions), [x2.shape[0], 1])
    return predictions


#######################################         k-folds cross validation         #######################################
def k_fold_cross_validation(k, x_data, y_data):
    conf_mat_sum = np.zeros((6, 6))
    precision_rate_sum = np.zeros((6))
    recall_rate_sum = np.zeros((6))
    f1_measure_rate_sum = np.zeros((6))
    class_rate_sum = 0

    fold_size = int(np.floor(x_data.shape[0] / k))

    # for each part
    for i in range(k):
        # break data set into k equal(ish) parts
        x_train = np.append(x_data[0:i * fold_size], x_data[(i + 1) * fold_size:], axis=0)
        y_train = np.append(y_data[0:i * fold_size], y_data[(i + 1) * fold_size:], axis=0)

        x_test = x_data[i * fold_size:(i + 1) * fold_size, :]
        y_test = y_data[i * fold_size:(i + 1) * fold_size]

        # train tree for each emotion on k-1 parts
        tree_list = []
        for emotion in range(1, 7):
            bin_emotion = subset(x_train, y_train, emotion)
            dec_tree = decision_tree_learning(x_train, range(45), bin_emotion)
            tree_list.append(dec_tree)

        pred_all_emotions = testTrees3(tree_list, x_test)

        # calculate evaluation metrics
        conf_mat = confusion_matrix(pred_all_emotions, y_test)
        precision_rate = precision(conf_mat)
        recall_rate = recall(conf_mat)

        # add to sums (for average later)
        conf_mat_sum += conf_mat
        precision_rate_sum += precision_rate
        recall_rate_sum += recall_rate
        f1_measure_rate_sum += f1_measure(precision_rate, recall_rate)
        class_rate_sum += classification_rate(conf_mat)

    print("Average confusion matrix:\n", conf_mat_sum / k)
    print("Average precision: ", np.round(precision_rate_sum / k, 2))
    print("Average recall:    ", np.round(recall_rate_sum / k, 2))
    print("Average f1_measure:", np.round(f1_measure_rate_sum / k, 2))
    print("Average classification_rate: ", np.round(class_rate_sum / k, 2))


#######################################             confusion matrix            #######################################
def confusion_matrix(predicted, actual):
    # predefine the confusion matrix
    cmat = np.zeros((6, 6))

    # increment the matrix
    for i in range(len(predicted)):
        cmat[actual[i] - 1, predicted[i] - 1] += 1

    # return the confusion matrix
    return cmat


#######################################         recall        #######################################
def recall(conf_mnatrix):
    # initialize the recall rate array
    rate = np.zeros((6))

    # compute the recall rate for each class
    for i in range(6):
        if sum(conf_mnatrix[i, :]) == 0:
            rate[i] = conf_mnatrix[i, i] * 100
        else:
            rate[i] = conf_mnatrix[i, i] * 100 / (sum(conf_mnatrix[i, :]))

    # return the recall rate
    return rate


#######################################         precision        #######################################
def precision(conf_matrix):
    # initialize the precision rate array
    rate = np.zeros((6))

    # compute the precision rate for each class
    for i in range(6):
        if sum(conf_matrix[:, i]) == 0:
            rate[i] = conf_matrix[i, i] * 100
        else:
            rate[i] = conf_matrix[i, i] * 100 / (sum(conf_matrix[:, i]))

    # return the precision rate
    return rate


#######################################         F1 mesure        #######################################
def f1_measure(precision_rate, recall_rate):
    # initialize the Fa measure array
    rate = np.zeros((6))

    # compute the Fa measure for each class
    for i in range(6):
        if precision_rate[i] == 0 or recall_rate[i] == 0:
            rate[i] = 0
        else:
            rate[i] = 2 * ((precision_rate[i]/100 * recall_rate[i]/100) / (precision_rate[i]/100 + recall_rate[i]/100))*100

    # return the Fa measure
    return rate


#######################################         classification rate        #######################################
def classification_rate(conf_matrix):
    # compute the classification rate

    rate = sum(conf_matrix.diagonal()) / conf_matrix.sum()

    # return the classification rate
    return rate


#######################################         all evaluation        #######################################
def evaluate_predictions(predictions, actual):
    conf_mat = confusion_matrix(predictions, actual)
    precision_rate = precision(conf_mat)
    recall_rate = recall(conf_mat)
    f1_measure_rate = f1_measure(precision_rate, recall_rate)
    class_rate = classification_rate(conf_mat)

    print("Average confusion matrix:\n", conf_mat)
    print("Average precision:\n", np.round(precision_rate, 2))
    print("Average recall:\n", np.round(recall_rate, 2))
    print("Average f1_measure:\n", np.round(f1_measure_rate, 2))
    print("Average classification_rate: ", np.round(class_rate, 2))


#######################################         visualisation        #######################################
def draw_tree(root, filename):
    graph = pydot.Dot(graph_type='graph')
    draw_children(root, graph, 'Path 0')
    graph.write_png('results/{}.png'.format(filename))


def draw_children(parent, graph, route):
    if parent.kids == [0, 0]:
        # this is a leaf node
        return
    else:
        parent_text = route + '\nAttribute ' + str(parent.getOp())

        for i in [0, 1]:
            child = parent.kids[i]
            side = ['L', 'R'][i]
            if child.getLabel() == None:
                # child is not a leaf node
                child_text = route + side + '\nAttribute ' + str(child.getOp())
                child_node = pydot.Node(child_text)  # , shape='box')
            else:
                # child is leaf node
                if child.getLabel():
                    color = "#66ff66"  # green
                else:
                    color = "#ff6666"  # red
                child_node = pydot.Node(route + side + '\nPrediction ' + str(child.getLabel()),
                                        shape='box', style="filled", fillcolor=color)

            graph.add_node(child_node)
            edge = pydot.Edge(parent_text, child_node, label=str(i))
            graph.add_edge(edge)

            draw_children(child, graph, route + side)

    return


#######################################         pickle trees        #######################################
def save_tree(tree, filename):
    fileobject = open(filename, 'wb')
    pickle.dump(tree, fileobject)
    fileobject.close()


def load_tree(filename):
    fileobject = open(filename, 'rb')
    return pickle.load(fileobject)
