import numpy as np


def gradient_descent(f, df, x0, y0, learning_rate, lr_decay, num_iters):
    """
    Find a local minimum of the function f(x) using gradient descent:
    Until the number of iteration is reached, decrease the current x and y points by the
    respective partial derivative times the learning_rate.
    In each iteration, record the current function value in the list f_list.
    The function should return the minimizing argument (x, y) and f_list.

    :param f: Function to minimize
    :param df: Gradient of f i.e, function that computes gradients
    :param x0: initial x0 point
    :param y0: initial y0 point
    :param learning_rate: Learning rate
    :param lr_decay: Learning rate decay
    :param num_iters: number of iterations
    :return: x, y (solution), f_list (array of function values over iterations)
    """
    f_list = np.zeros(num_iters) # Array to store the function values over iterations
    x, y = x0, y0
    # TODO: Implement the gradient descent algorithm with a decaying learning rate
    pass

    return x, y, f_list


def ackley(x, y):
    """
    Ackley function at point (x, y)
    :param x: X coordinate
    :param y: Y coordinate
    :return: f(x, y) where f is the Ackley function
    """
    # TODO: Implement the Ackley function (as specified in the Assignment 1 sheet)
    return None


def gradient_ackley(x, y):
    """
    Compute the gradient of the Ackley function at point (x, y)
    :param x: X coordinate
    :param y: Y coordinate
    :return: \nabla f(x, y) where f is the Ackley function
    """
    # TODO: Implement partial derivatives of Ackley function w.r.t. x and y
    df_dx = None
    df_dy = None

    gradient = np.array([df_dx, df_dy])
    return gradient
