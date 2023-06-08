import numpy as np

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def sigmoid(self, x):
        t = np.dot(x, self.theta)
        ePow = np.e ** -t

        return 1 / (1 + ePow)
    
    def loss_function(self, x, y):
        lhs = np.dot(-y , np.log(self.sigmoid(x)))
        rhs = np.dot((1 - y) , np.log(1 - self.sigmoid(x)))

        return (lhs - rhs) / len(y)

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.theta = np.ones(X.shape[1] + 1)
        # Bias trick:
        col = np.ones(len(X))  
        X_trick = np.c_[col, X]

        for i in range(self.n_iter):
          h = np.dot(X_trick.T ,self.sigmoid(X_trick) - y)
          self.theta = self.theta - self.eta * h
          loss = self.loss_function(X_trick, y)
          self.thetas.append(self.theta)
          if i > 0 and (np.abs(self.Js[-1] - loss) < self.eps):
              break
          self.Js.append(loss)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        col = np.ones(len(X))
        X = np.c_[col, X]

        for row in X:
            sigmoid = self.sigmoid(row)
            if sigmoid > 0.5:
                preds.append(1)
            else :
                preds.append(0)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return np.array(preds)

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    SumAccuracies = 0
    shuffledIndices = np.arange(len(X))
    np.random.shuffle(shuffledIndices)
    X = X[shuffledIndices]
    y = y[shuffledIndices]

    foldSize = len(X) // folds # 5-fold cross validation

    foldsX = []
    foldsY = []
    index = 0

    for k in range(folds):
      foldsX.append(X[index : index + foldSize])
      foldsY.append(y[index : index + foldSize])
      index += foldSize
 
    for i in range(folds):
      X_valid = foldsX[i]
      y_valid = foldsY[i]

      for j in range(folds):
        initiated = False
        if j == i:
          continue
        if initiated == False:
          X_train = foldsX[j]
          y_train = foldsY[j]
          initiated = True
        else:
          X_train = np.concatenate(foldsX[j])
          y_train = np.concatenate(foldsY[j]) 
        
      algo.fit(X_train, y_train)
      preds = algo.predict(X_valid)
      SumAccuracies += np.count_nonzero(preds == y_valid) / len(y_valid)
            
    cv_accuracy = SumAccuracies / folds       
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    p = (1 / np.sqrt(2 * np.pi * (sigma ** 2))) * np.e ** -(((data - mu) ** 2) / (2 * (sigma ** 2)))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.responsibilities = np.ones((len(data), self.k)) / self.k
        self.weights = np.ones(self.k) / self.k
        self.mus = np.random.uniform(np.min(data), np.max(data), self.k)
        self.sigmas = np.random.rand(self.k)
        self.costs = []
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        numerator = self.weights * norm_pdf(data, self.mus, self.sigmas)
        denominator = np.sum(numerator, axis = 1, keepdims=True)
        self.responsibilities = numerator / denominator
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.weights = np.sum(self.responsibilities, axis=0) / len(self.responsibilities)
        self.mus = np.sum(self.responsibilities * data, axis=0) / (len(self.responsibilities) * self.weights)
        rhs = (data - self.mus) ** 2
        self.sigmas = np.sqrt(np.sum(self.responsibilities * rhs, axis=0) / (len(self.responsibilities) * self.weights))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def cost(self, data):
        cost = 0
        
        for i in range(len(data)):
            for j in range(self.k):
                cost -= np.log(self.weights[j] * norm_pdf(data[i], self.mus[j], self.sigmas[j]))
            
        return cost

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.init_params(data)
        for i in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)
            cost = self.cost(data)
            if i > 1 and np.abs(self.costs[-1] - cost) < self.eps:
                break
            self.costs.append(cost)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pdf = np.zeros_like(data)

    for i in range(len(weights)):
        pdf += weights[i] * norm_pdf(data, mus[i], sigmas[i])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.em_models = []

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.classes, classesCounts = np.unique(y, return_counts=True)
        self.prior = np.empty(len(self.classes))
        for i, classif in enumerate(self.classes):
            self.prior[i] = classesCounts[i] / len(y)
            indicies = np.where(y == classif)[0]
            X_class = X[indicies]
            em = EM(self.k)
            em.fit(X_class.reshape(-1, 1))
            self.em_models.append(em)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for x in X:
            class_scores = []
            for i, c in enumerate(self.classes):
                class_score = self.prior[i]
                em = self.em_models[i]
                dist_params = em.get_dist_params()
                for feature in x.T:
                    pdf = gmm_pdf(feature, dist_params[0], dist_params[1], dist_params[2])
                    class_score *= pdf
                class_scores.append(class_score)

            # Normalize probs
            sum_probs = np.sum(class_scores)
            class_scores = class_scores / sum_probs
            preds.append(self.classes[np.argmax(class_scores)])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return np.array(preds)
    
def calc_accuracy(preds, y_truth):
    correct_count = 0
    for true_label, pred_label in zip(preds, y_truth):
        if true_label == pred_label:
            correct_count += 1

    accuracy = correct_count / len(y_truth)
    return accuracy
        

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    lor = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor.fit(x_train, y_train)
    nbg = NaiveBayesGaussian(k)
    nbg.fit(x_train, y_train)

    lor_test_preds = lor.predict(x_test)
    lor_train_preds = lor.predict(x_train)
    lor_test_acc = calc_accuracy(lor_test_preds, y_test)
    lor_train_acc = calc_accuracy(lor_train_preds, y_train)

    nbg_test_preds = nbg.predict(x_test)
    nbg_train_preds = nbg.predict(x_train)
    bayes_train_acc = calc_accuracy(nbg_train_preds, y_train)
    bayes_test_acc = calc_accuracy(nbg_test_preds, y_test)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }