# import necessary libraries

import sys

import matplotlib.pyplot as plt
import pandas as pd

# import necessary functions

sys.path.append("/home/excviral/Pycharm/PycharmProjects/Adaptive-forex-forecast/Adaptive filters/")
sys.path.append("/home/excviral/Pycharm/PycharmProjects/Adaptive-forex-forecast/Feature extractors/")

from lms import lms
from feature_extractior_functions import simple_amv_extractor
from normalization_functions import simple_normalize


# =====================================================================================================================
# ===== Helper functions ==============================================================================================
# =====================================================================================================================

def predictor(inputs, weights):
    return sum([i * j for i, j in zip(inputs, weights)])


# =====================================================================================================================
# ===== Forecast ======================================================================================================
# =====================================================================================================================

# import dataset
dataset = pd.read_csv('data.csv')
data = list(dataset['Price'].values)

# normalizing dataset
n_data = simple_normalize(data)
# print(len(n_data),n_data)

# define window size
window_size = 10

# feature extraction
feature_table = simple_amv_extractor(window_size, n_data[:(len(n_data) - 1)])
# print(len(feature_table), feature_table)

# seperating training data and test data
training_data = feature_table[:int(len(feature_table) * 0.8)]
print(len(training_data), training_data)
testing_data = feature_table[int(len(feature_table) * 0.8):]
print(len(testing_data), testing_data)


# start training
def training(training_data):
    # list for storing weights
    weights = [0, 0, 0]
    # list for storing errors
    errors = []
    for i in training_data:
        # print("******")
        x_k = i[:len(i) - 1]
        # print(x_k)
        y_k = predictor(x_k, weights)
        # print(y_k)
        d_k = i[len(i) - 1]
        # print(d_k)
        e_k = d_k - y_k
        # print(e_k)
        errors.append(e_k)
        weights = lms(weights, 0.000195, x_k, e_k)
        # print(weights)
    plt.plot(errors)
    plt.show()
    return weights


weights = training(training_data)

print(weights)


# start testing
def testing(testing_data, weights):
    errors = []
    desired = []
    predicted = []
    for i in testing_data:
        # print("******")
        x_k = i[:len(i) - 1]
        # print(x_k)
        y_k = predictor(x_k, weights)
        predicted.append(y_k)
        # print(y_k)
        d_k = i[len(i) - 1]
        desired.append(d_k)
        # print(d_k)
        e_k = d_k - y_k
        # print(e_k)
        errors.append(e_k)
    plt.plot(errors)
    plt.show()

    print(desired)
    print(predicted)

    plt.plot(desired, label="Desired Values")
    plt.plot(predicted, label="Predicted Values")
    plt.show()


testing(testing_data, weights)
