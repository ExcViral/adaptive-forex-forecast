# This script contains functions that will be helpful for normalizing dataset

# There are numerous techniques available for normalizing the data in statistics, however I will be implementing
# basic ones for now, later on I will implement other normalization techniques in this script.

# Simplest normalization technique: Divide each value in the dataset with the maximum value in the dataset.
# TODO: Other normalization technique: (X-mu)/sigma

def simple_normalize(data):
    '''
    Simplest normalization function

    This function takes in a list, finds the maximum element in the list and divides all the elements of the list by
    maximum element, and returns the list with maximum element appended to the end of list. As a result, the data is
    normalized between 0 and 1.

    :param data: (list) containing features to be normalized
    :return: (list) normalized features + maximum value
    '''
    max_elem = max(data)
    normalized_data = [i / max_elem for i in data]
    normalized_data.append(max_elem)
    return normalized_data

