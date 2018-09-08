# LMS algorithm belongs to the class of adaptive filters.
# The basic goal of the LMS algorithm is to find the optimum weights so as the solution converges to optimum result.
# My forex prediction model will be using this algorithm in order to optimize its weights and make best prediction.

# The LMS algorithm can be represented mathematically in one equation:
# W[k+1] = W[k] + 2*mu*e(k)*X[k]

# =====================================================================================================================
# ===== Helper functions ==============================================================================================
# =====================================================================================================================

def add_lists(a, b):
    '''
    This function performs element wise addition of two lists a and b, a naive implementation of vector addition.

    :param a: (list) vector a
    :param b: (list) vector b
    :return: (list) vector (a+b)
    '''
    if len(a) == len(b):
        return [i + j for i, j in zip(a, b)]
    else:
        raise ValueError('Incompatible list widths')


def sub_lists(a, b):
    '''
    This function performs element wise subtraction of two lists a and b, a naive implementation of vector subtraction.

    :param a: (list) vector a
    :param b: (list) vector b
    :return: (list) vector(a-b)
    '''
    if len(a) == len(b):
        return [i + j for i, j in zip(a, b)]
    else:
        raise ValueError('Incompatible list widths')


def scalar_mul(a, b):
    '''
    This function performs scalar multiplication of a with vector b.

    :param a: (float) scalar a
    :param b: (list) vector b
    :return: (list) scalar multiplication of a with b
    '''
    return [a * i for i in b]


# =====================================================================================================================
# ===== LMS Algorithm =================================================================================================
# =====================================================================================================================

def lms(current_weights, mu, current_input, current_error):
    '''
    A naive implementation of lms algorithm which calculates new weights based on input parameters.

    This function computes new weights W[k+1] using the equation:
    W[k+1] = W[k] + 2*mu*e(k)*X[k]

    :param current_weights: (list) W[k] in the above equation, current weights
    :param mu: (float) learning parameter, typically 0 < mu < 1
    :param current_input: (float) X[k] in the above equation, current input
    :param current_error: (float) e(k) in the above equation, current error
    :return: (list) updated weights, W[k+1] in the above equation
    '''
    delta_weight = scalar_mul((2 * mu * current_error), current_input)
    new_weights = add_lists(current_weights, delta_weight)
    return new_weights
