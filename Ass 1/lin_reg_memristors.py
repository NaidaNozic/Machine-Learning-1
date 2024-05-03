from enum import Enum
from typing import Tuple
import numpy as np


class MemristorFault(Enum):
    IDEAL = 0
    DISCORDANT = 1
    STUCK = 2
    CONCORDANT = 3



def model_to_use_for_fault_classification():
    return 2 # TODO: change this to either 1 or 2 (depending on which model you decide to use)


def fit_zero_intercept_lin_model(x: np.ndarray, y: np.ndarray) -> float:
    """
    :param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    :param y: y coordinates of data points (i.e., \Delta R_i$)
    :return: theta 
    """

    # Implemented equation for theta containing sums
    theta = np.sum(x * y) / np.sum(x**2)
    return theta


def bonus_fit_lin_model_with_intercept_using_pinv(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    :param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    :param y: y coordinates of data points (i.e., \Delta R_i$)
    :return: theta_0, theta_1
    """
    from numpy.linalg import pinv
    # Constructing the design matrix 'X' that includes a constant column for the intercept
    num_samples = len(x)  # Determine the number of samples in the dataset
    X = np.column_stack((np.ones(num_samples), x))  # Design matrix with a column of ones and x values

    # Calculating the pseudoinverse of the design matrix 'X'
    X_pseudoinverse = pinv(X)  # Generate the Moore-Penrose pseudoinverse

    # Computing the coefficient vector 'theta' through matrix operations
     
    # TODO- DONE: implement the equation for theta using the pseudo-inverse (Bonus Task)
    theta = X_pseudoinverse @ y  # Using '@' for matrix multiplication
    return theta[0], theta[1]


def fit_lin_model_with_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    :param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    :param y: y coordinates of data points (i.e., \Delta R_i$)
    :return: theta_0, theta_1 
    """

    # TODO - DONE: implement the equation for theta_0 and theta_1 containing sums

    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x**2)
    
    denominator = n * sum_x2 - sum_x**2
    theta_1 = (n * sum_xy - sum_x * sum_y) / denominator
    theta_0 = (sum_y - theta_1 * sum_x) / n

    return theta_0, theta_1


def classify_memristor_fault_with_model1(theta: float) -> MemristorFault:
    """
    :param theta: the estimated parameter of the zero-intercept linear model
    :return: the type of fault
    """
    #       Implement either this function, or the function `classify_memristor_fault_with_model2`,
    #       depending on which model you decide to use.

    # If you decide to use this function, remove the line `raise NotImplementedError()` and
    # return a MemristorFault based on the value of theta.
    # For example, return MemristorFault.IDEAL if you decide that the given theta does not indicate a fault, and so on.
    # Use if-statements and choose thresholds for the parameters that make sense to you.

    

    raise NotImplementedError()


def classify_memristor_fault_with_model2(theta0: float, theta1: float) -> MemristorFault:
    """
    :param theta0: the intercept parameter of the linear model
    :param theta1: the slope parameter of the linear model
    :return: the type of fault
    """
    # TODO: Implement either this function, or the function `classify_memristor_fault_with_model1`,
    #       depending on which model you decide to use.

    # If you decide to use this function, remove the line `raise NotImplementedError()` and
    # return a MemristorFault based on the value of theta0 and theta1.
    # For example, return MemristorFault.IDEAL if you decide that the given theta pair
    # does not indicate a fault, and so on.
    # Use if-statements and choose thresholds for the parameters that make sense to you.
    if theta1 > 0 and abs(theta1 - 1.0) < 0.3:
        return MemristorFault.IDEAL
    elif theta1 < 0 and abs(theta1) > 0.3:
        return MemristorFault.DISCORDANT
    elif (theta1 >= 0.3 and theta1 <= 0.7) or (theta1 > 1.3):
        return MemristorFault.CONCORDANT
    else:
        return MemristorFault.STUCK
    #raise NotImplementedError()
