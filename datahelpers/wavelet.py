import numpy as np
import pywt as pw


def interpolate_table(input_list, return_size):
    """
    interpolate_table(input_list, return_size)
    expands a list A of size M to a list B of size N
    using stepwise approximation
    list A defines a function f(x) = f_i, x_i <= x x_{i+1},
    where x_i = i / len(A), i = 0..M-1

    args:
    input_list -

    return_size -

    returns:

    """

    def step_function(x):
        """stepwise function approximation to the input_list on [0, 1]
        """
        k = int(x * len(input_list))
        return input_list[k]

    vfunc = np.vectorize(step_function)
    x = np.arange(float(return_size)) / return_size
    r = vfunc(x)
    r *= (sum(input_list) / len(input_list)) / (sum(r) / return_size)
    return r


def low_freq_wavedec(input_list, order=10, order_low_frequency=4):
    """
    NB:


    args:

    returns:

    pw.waverec(return_, 'haar') gets back the approximation of the original


    """
    input_ = interpolate_table(input_list, 2 ** order)
    spec = pw.wavedec(input_, "haar")[: order_low_frequency + 1]
    conv_const = 2 ** (0.5 * (order_low_frequency - order))
    return [s * conv_const for s in spec]
