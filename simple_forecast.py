# import necessary libraries

import sys

import matplotlib.pyplot as plt
import pandas as pd

# import necessary modules

sys.path.append("/home/excviral/Pycharm/PycharmProjects/Adaptive-forex-forecast/Adaptive filters/")
sys.path.append("/home/excviral/Pycharm/PycharmProjects/Adaptive-forex-forecast/Feature extractors/")

from lms import lms
from feature_extractior_functions import simple_amv_extractor
from normalization_functions import simple_normalize

# import dataset
dataset = pd.read_csv('data.csv')
data = list(dataset['Price'].values)

# =====================================================================================================================
# ===== Helper functions ==============================================================================================
# =====================================================================================================================

def predict(inputs, weights):
    '''
    Model to predict the exchange rate based on inputs and weights

    This function takes in features as input, in this case the features are actual value, mean value, and variance, and
    then the features are multiplied with its respective weight and added up to compute the predicted value.
    Mathematically: y_k = a*w0 + m*w1 + v*w2

    :param inputs: (list) containing features of the dataset
    :param weights: (list) containing weights for each feature of the dataset
    :return: (float) predicted outcome based on input and weights
    '''
    return sum([i * j for i, j in zip(inputs, weights)])


def plot_convergence_characteristics(errors):
    '''
    Function to plot the convergence characteristics viz. plot of (error)^2 v/s pattern number

    Convergence characteristics shows the rate of convergence of the error of prediction v/s actual value, to 0.

    :param errors: (list) containing errors corresponding to each pattern
    :return: none
    '''
    errors_squared = [i * i for i in errors]
    plt.plot(errors_squared)
    plt.xlabel('Pattern number')
    plt.ylabel('(Error)^2')
    plt.show()


# =====================================================================================================================
# ===== Forecast Algorithm ============================================================================================
# =====================================================================================================================

# first we normalize the data, we use simple normalization technique of dividing each value in data by max value of data
n_data = simple_normalize(data)

# define window size
window_size = 10

# extract the feature-patterns (a,m,v,tv) from the data, and store it in a list
feature_table = simple_amv_extractor(window_size, n_data[:(len(n_data) - 1)])

# We will be using 80% of the feature-patterns from the feature table for training the model, and the remaining 20% of
# the feature-patters will be used for testing the model

# separating training data and test data and storing them in respective lists for later use
training_data = feature_table[:int(len(feature_table) * 0.8)]
testing_data = feature_table[int(len(feature_table) * 0.8):]


# Now that the data is ready, we train our model to compute optimum weights

# Model trainer
def train_model(training_data, mu):
    '''
    This function trains the prediction model to find the optimum weights for which prediction error is minimum

    Algorithm: Initially the model starts with weights = [0,0,0], the model then predicts some value, then it computes
    error vs the actual value/desired value and adjusts the weights accordingly. This is repeated until all patterns in
    the training set are exhausted. At the end, the error should have converged to zero, this can be seen in the
    convergence characteristics plot generated at the end of training.
    NOTE: The convergence of error is highly dependent on the choice of 'mu' - the convergence coefficient.
    Theoretically, its value should lie between 0 and 1, when mu is closer towards zero learning rate will be slow,
    but accuracy will be more, when it is closer to 1, learning rate will be faster, but accuracy will be poor.

    :param training_data: (list) containing features selected to train the model
    :param mu: (float) convergence coeffecient
    :return: (list) containing optimized weights, which can be used for prediction
    '''

    # This list will store weights, initially the weights will be zeros
    weights = [0, 0, 0]
    # This list will store errors corresponding to each pattern
    errors = []

    # This loop optimizes the weights, such that error converges to zero
    for i in training_data:
        # Inputs to the predictor model [a, m, v]
        x_k = i[:len(i) - 1]
        # Desired value or target value, that is to be predicted
        d_k = i[len(i) - 1]
        # Predict the output price based on the input and current weights
        y_k = predict(x_k, weights)
        # Compare the predicted price to the desired price, and compute the error
        e_k = d_k - y_k
        # Store the error to the list
        errors.append(e_k)
        # Compute new weights based on the previous weights, mu, previous input, previous error using lms algorithm
        weights = lms(weights, mu, x_k, e_k)

    plot_convergence_characteristics(errors)
    return weights


weights = train_model(training_data, 0.000195)


# Now that we have computed the optimum weights, we test it against the feature-patterns that we had stored earlier

# start testing
def test_model(testing_data, weights):
    '''
    This function tests the performance of the model so that we can know the accuracy of the model

    :param testing_data: (list) containing features that are to be tested
    :param weights: (list) containing optimum weights generated during training
    :return: none
    '''

    # Lists to store errors, desired/target value, and predicted value for each test-pattern
    errors = []
    desired = []
    predicted = []

    # This loop predicts a value for each pattern, computes error against desired value and stores them in lists above
    for i in testing_data:
        # Inputs to the predictor model
        x_k = i[:len(i) - 1]
        # Desired value or target value, that is to be predicted
        d_k = i[len(i) - 1]
        # Predict the output price based on the input and optimum weights generated during training
        y_k = predict(x_k, weights)
        # Store the predicted value and desired value to the respectivs lists
        predicted.append(y_k)
        desired.append(d_k)
        # Compute the error of prediction against the desired value
        e_k = d_k - y_k
        # Store the error to the list
        errors.append(e_k)

    # Plot the predicted and desired values to compare the error
    plt.plot(desired, 'g-', label="Desired Values")
    plt.plot(predicted, 'r-', label="Predicted Values")
    plt.xlabel('Pattern number')
    plt.ylabel('Normalized Exchange rate')
    plt.legend(loc='best')
    plt.show()


test_model(testing_data, weights)

# To predict exchange rate against a new feature after training, simply plug the features into 'predict' function
