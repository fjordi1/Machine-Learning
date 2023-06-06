###### Your ID ######
# ID1: 207441635
# ID2: 206174468
#####################

# imports
from json.encoder import INFINITY
from operator import indexOf
from tkinter.tix import MAX
from jupyterlab_server import app
import numpy as np
import pandas as pd


def preprocess(X, y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################
    X = (X - np.mean(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    y = (y - np.mean(y, axis=0)) / (np.max(y, axis=0) - np.min(y, axis=0))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.                             #
    ###########################################################################
    col = np.ones(len(X))  # X.shape[0]
    X = np.c_[col, X]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """

    J = 0  # We use J for the cost.
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    ###########################################################################
    m = len(X)
    h = np.dot(X, theta)  # X.dot(theta)
    errors = h - y

    J = np.sum(errors ** 2) / (m * 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    for i in range(num_iters):
        m = len(X)
        h = np.dot(X, theta)
        errors = h - y
        sum = alpha * (np.dot(errors, X)) / m

        theta = theta - sum
        J_history.append(compute_cost(X, y, theta))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """

    pinv_theta = []
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    Xt_Xinv = np.linalg.inv(np.matmul(X.transpose(), X))
    pinv = np.matmul(Xt_Xinv, X.transpose())
    pinv_theta = np.matmul(pinv, y)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the efficient gradient descent optimization algorithm.  #
    ###########################################################################
    for i in range(num_iters):
        m = len(X)
        h = np.dot(X, theta)
        errors = h - y
        sum = alpha * (np.dot(errors, X)) / m

        theta = theta - sum
        loss = compute_cost(X, y, theta)
        if (i > 0 and (J_history[-1] - loss) <= 1e-8):
            break
        J_history.append(loss)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """

    alphas = [0.00001, 0.00003, 0.0001, 0.0003,
              0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}  # {alpha_value: validation_loss}
    ###########################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    for alpha in alphas:
        theta = np.ones(X_train.shape[1])
        gradDes_theta, loss = efficient_gradient_descent(
            X_train, y_train, theta, alpha, iterations)
        validation_loss = compute_cost(X_val, y_val, gradDes_theta)
        alpha_dict[alpha] = validation_loss
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    ##### c######################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    X_train = apply_bias_trick(X_train)
    X_val = apply_bias_trick(X_val)
    selected_features.append(0)
    for i in range(min(5, X_train.shape[1] - 1)):
        bestError = INFINITY
        bestFeature = 0
        # shape[1] = number of columns
        for j in range(X_train.shape[1]):
            if (j in selected_features):
                continue
            selected_features.append(j)
            theta = np.ones(len(selected_features))

            # Fit the training and validation data to the selected features
            X_train_fit = X_train[:, selected_features]
            X_val_fit = X_val[:, selected_features]

            # Create the model
            theta, error = efficient_gradient_descent(
                X_train_fit, y_train, theta, best_alpha, iterations)
            # Compute the error for that model
            error = compute_cost(X_val_fit, y_val, theta)
            if (error < bestError):
                bestFeature = j
                bestError = error

            selected_features.pop()
        selected_features.append(bestFeature)  # Append permanently
    # Remove the bias column from the selected features
    selected_features.pop(0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    poly_dict = pd.DataFrame.to_dict(df_poly)
    for (column1, column1Data) in df_poly.items():
        for (column2, column2Data) in df_poly.items():
            if (column1 == column2):
                name = column1 + '^2'
            else:
                name = column1 + '*' + column2
            flippedName = column2 + '*' + column1
            if flippedName not in poly_dict:
                poly_dict[name] = column1Data * column2Data
    df_poly = pd.DataFrame.from_dict(poly_dict)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly
