import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    sumPoisonous = 0
    sumEdible = 0

    for entry in data[:, -1]:
        if entry == 'e':
            sumEdible += 1
        else:
            sumPoisonous += 1

    totalEntries = sumPoisonous + sumEdible

    gini += 1 - ((sumEdible / totalEntries)**2 + (sumPoisonous / totalEntries)**2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    sumPoisonous = 0
    sumEdible = 0

    for entry in data[:, -1]:
        if entry == 'e':
            sumEdible += 1
        else:
            sumPoisonous += 1

    totalEntries = sumPoisonous + sumEdible
    probEdible = sumEdible / totalEntries
    probPoisonous = sumPoisonous / totalEntries

    if probEdible != 0:
        entropy += probEdible * np.log2(probEdible)
    if probPoisonous != 0:
        entropy += probPoisonous * np.log2(probPoisonous)

    entropy *= -1
    entropy = np.abs(entropy)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0
    groups = {} # groups[feature_value] = data_subset
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    uniqueArrays = np.unique(data[:, feature], return_counts=True)
    totalEntries = len(data)
    baseImpurity = impurity_func(data)

    for i, value in enumerate(uniqueArrays[0]):
        mask = (data[:, feature] == value)
        groups.update({value : data[mask, :]})
        impurity = impurity_func(data[mask, :])
        probabilityOfValue = uniqueArrays[1][i] / totalEntries
        goodness += probabilityOfValue * impurity

    goodness = baseImpurity - goodness

    if gain_ratio == True and goodness != 0:
        splitInInformation = 0
        for count in uniqueArrays[1]:
            probability = count / totalEntries
            splitInInformation += probability * np.log2(probability)
        goodness = goodness_of_split(data, feature, calc_entropy)[0] / (-1 * splitInInformation)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return goodness, groups

def countLabels(data):
    sumEdible = 0
    sumPoisonous = 0
    for entry in data[:, -1]:
        if entry == 'e':
            sumEdible += 1
        else:
            sumPoisonous += 1
    return sumEdible, sumPoisonous

def most_common_label(data):
    sumEdible, sumPoisonous = countLabels(data)
    return 'p' if sumPoisonous > sumEdible else 'e'

class DecisionNode:

    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio
        self.alreadyUsedFeatures = []

    
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pred = most_common_label(self.data)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
     
    def split(self, impurity_func):

        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.depth == self.max_depth:
            return
        
        bestSplit = 0
        bestSplitIndex = -1
        for i in range(self.data.shape[1] - 1):
            split = goodness_of_split(self.data, i, impurity_func, self.gain_ratio)[0]
            if split > bestSplit and i not in self.alreadyUsedFeatures:
                bestSplit = split
                bestSplitIndex = i
        
        if bestSplitIndex == -1:
            return

        childList = []
        if self.chi != 1:
            ChiSquareStatistic = 0
            probabilityEdible, probabilityPoison = countLabels(self.data)
            probabilityEdible /= len(self.data)
            probabilityPoison /= len(self.data)

        valuesArray = np.unique(self.data[:, bestSplitIndex])
        for value in valuesArray:
            mask = (self.data[:, bestSplitIndex] == value)
            onlyValueRows = self.data[mask, :]
            node = DecisionNode(onlyValueRows, bestSplitIndex, self.depth + 1, self.chi, self.max_depth, self.gain_ratio)
            node.terminal = True
            node.alreadyUsedFeatures = self.alreadyUsedFeatures.copy()
            node.alreadyUsedFeatures.append(bestSplitIndex)
            if self.chi != 1:
                ChiSquareStatistic += chi_square_test(onlyValueRows, probabilityEdible, probabilityPoison)
            childList.append(node)

        if self.chi != 1 and ChiSquareStatistic < chi_table[len(valuesArray) - 1][self.chi]:
            return

        for i, value in enumerate(valuesArray):
            self.add_child(childList[i], value)
        
        self.feature = bestSplitIndex
        self.terminal = False
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

def chi_square_test(onlyValueRows, probabilityEdible, probabilityPoison):   #probabilityEdible, ProbabilityPoison, Df, Pf, Nf):
    Df = len(onlyValueRows)
    Pf, Nf = countLabels(onlyValueRows)
    EZero = Df * probabilityEdible
    EOne = Df * probabilityPoison
    return (((Pf - EZero)**2) / EZero) + (((Nf - EOne)**2) / EOne)


def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    root = DecisionNode(data, -1, 0, chi, max_depth, gain_ratio)
    queue = []
    queue.append(root)
    while(len(queue) > 0):
        currentNode = queue.pop(0)
        if(len(np.unique(currentNode.data[:, -1])) == 1):
            continue
        currentNode.split(impurity)
        for child in currentNode.children:
            queue.append(child)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root

def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    currentNode = root
    while(currentNode.terminal != True):
        instanceFeatureValue = instance[currentNode.feature]
        try:
            indexOfChild = currentNode.children_values.index(instanceFeatureValue)
        except:
            pred = 'p'
            return pred
        currentNode = currentNode.children[indexOfChild]

    pred = currentNode.pred
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred

def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    sumAccurate = 0
    for row in dataset:
        if predict(node, row) == row[-1]:
            sumAccurate += 1

    accuracy = (sumAccurate / len(dataset)) * 100
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy

def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing  = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = build_tree(X_train, calc_entropy, True, 1, max_depth)
        training.append(calc_accuracy(tree, X_train))
        testing.append(calc_accuracy(tree, X_test))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return training, testing


def chi_pruning(X_train, X_test):

    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for chi in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        tree = build_tree(X_train, calc_entropy, True, chi)
        chi_training_acc.append(calc_accuracy(tree, X_train))
        chi_testing_acc.append(calc_accuracy(tree, X_test))
        depth.append(tree.max_depth)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return chi_training_acc, chi_testing_acc, depth

def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    n_nodes = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    sumNodes = 0
    queue = []
    queue.append(node)
    while (len(queue) > 0) :
        currentNode = queue.pop()
        sumNodes += 1
        for child in currentNode.children:
            queue.append(child)

    n_nodes = sumNodes
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






