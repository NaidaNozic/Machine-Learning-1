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
    # TODO - DONE: Implement the gradient descent algorithm with a decaying learning rate
    for i in range(num_iters):
        # Calculate gradients for both coordinates
        dx, dy = df(x, y)

        # Update the coordinates by moving against the gradient
        x -= learning_rate * dx
        y -= learning_rate * dy

        # Record the function value at the current coordinates
        f_list[i] = f(x, y)

        # Apply decay to the learning rate
        learning_rate *= lr_decay

    return x, y, f_list


def ackley(x, y):
    """
    Ackley function at point (x, y)
    :param x: X coordinate
    :param y: Y coordinate
    :return: f(x, y) where f is the Ackley function
    """
    # TODO - DONE: Implement the Ackley function (as specified in the Assignment 1 sheet)
    # First part of the function involving exponentiation
    part1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    # Second part of the function involving cosines and exponentiation
    part2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    # Final Ackley function value
    value = part1 + part2 + np.e + 20
    return value


def gradient_ackley(x, y):
    """
    Compute the gradient of the Ackley function at point (x, y)
    :param x: X coordinate
    :param y: Y coordinate
    :return: \nabla f(x, y) where f is the Ackley function
    """
    # TODO- DONE: Implement partial derivatives of Ackley function w.r.t. x and y
    # Derived expressions for partial derivatives of the Ackley function w.r.t. x and y
    exp_component = np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    cos_component = np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))

    # Partial derivative with respect to x
    df_dx = 2 * exp_component * (x / np.sqrt(0.5 * (x**2 + y**2))) + np.pi * cos_component * np.sin(2 * np.pi * x)

    # Partial derivative with respect to y
    df_dy = 2 * exp_component * (y / np.sqrt(0.5 * (x**2 + y**2))) + np.pi * cos_component * np.sin(2 * np.pi * y)

    gradient = np.array([df_dx, df_dy])
    return gradient
