# This script contains the functions that will be helpful for extracting features from the dataset

# Import necessary libraries

import math


# =====================================================================================================================
# ===== Feature extractor functions ===================================================================================
# =====================================================================================================================

def simple_amv_extractor(window_size, data):
    '''
    A simple feature extraction function that extracts actual value, mean value, and variance

    This function takes in a list and a window size, and starting from the first member of the list, it calculates the
    actual value 'a', mean of the window 'm', and variance of the window 'v', and the target value 't' and then the
    window slides, calculation is done again, and so on till the end of the list is reached, and a feature table is
    generated.
    Illustrative explanation of function working:
        eg. dataset = [x1, x2, x3, x4, x5, x6, x7], window size = 4
        #1 Iteration: a = x4, m = (x1+x2+x3+x4)/4, v = sqrt(((x1-x2)^2)+((x1-x3)^2)+((x1-x4)^2)), t = x5
        #2 Iteration: a = x5, m = (x2+x3+x4+x5)/4, v = sqrt(((x2-x3)^2)+((x2-x4)^2)+((x2-x5)^2)), t = x6
        #3 Iteration: a = x6, m = (x3+x4+x5+x6)/4, v = sqrt(((x3-x4)^2)+((x3-x5)^2)+((x3-x6)^2)), t = x7

    :param window_size: (int) # of elements in a window
    :param data: (list) Input data, for forex prediction - exchange rates
    :return: (list) containing features - a, m, v, t
    '''

    # list to store features
    feature_table = []
    # looping through the dataset to extract features
    for i in range(len(data) - window_size):

        feature = []
        a = data[i + window_size - 1]
        m = 0
        v = 0
        t = data[i + window_size]

        for j in range(window_size):

            d = data[i + j]
            m = m + d
            if (j > 0):
                v = v + pow((data[i] - d), 2)

        m = m / window_size
        v = math.sqrt(v)

        feature.append(a)
        feature.append(m)
        feature.append(v)
        feature.append(t)

        feature_table.append(feature)

    return feature_table
