import numpy as np
from sklearn.base import BaseEstimator


def loss(w, b, C, X, y):
    # Implement the loss function (Equation 1)
    # useful methods: np.sum, np.clip
    regularization_term = w @ w * 0.5
    hinge_loss = np.maximum(0, 1 - y * (X @ w + b))
    total_loss = regularization_term + C * np.sum(hinge_loss)
    
    return total_loss


def grad(w, b, C, X, y):
    # Implement the gradients of the loss with respect to w and b.
    # Useful methods: np.sum, np.where, numpy broadcasting
    margin = 1 - y * ((X @ w) + b)

    wrt_b_value = np.where(margin > 0, y, 0)
    grad_b = - C * np.sum(wrt_b_value)

    wrt_w_value = np.where(margin > 0, 1, 0)
    grad_w = w - C * np.sum(np.column_stack((y, y)) * X * np.column_stack((wrt_w_value, wrt_w_value)))

    return grad_w, grad_b


class LinearSVM(BaseEstimator):
    def __init__(self, C=1, eta=1e-3, max_iter=1000):
        self.C = C
        self.max_iter = max_iter
        self.eta = eta
        self.w = None
        self.b = None

    def fit(self, X, y):
        # convert y such that components are not \in {0, 1}, but \in {-1, 1}
        y = np.where(y == 0, -1, 1)

        # TODO: Initialize self.w and self.b. Does the initialization matter?
        self.w = np.zeros(X.shape[1])
        self.b = 0

        loss_list = []
        eta = self.eta  # starting learning rate
        for j in range(self.max_iter):
            # Compute the gradients, update self.w and self.b using `eta` as the learning rate.
            # Compute the loss and add it to loss_list.
            grad_w, grad_b = grad(self.w, self.b, self.C, X, y)

            self.w -= eta * grad_w
            self.b -= eta * grad_b

            loss_list.append(loss(self.w, self.b, self.C, X, y))
            # decaying learning rate
            eta = eta * 0.99

        return loss_list

    def predict(self, X):
        # Predict class labels of unseen data points on rows of X
        # The output should be a vector of 0s and 1s (*not* -1s and 1s)
        y_pred = X @ self.w + self.b
        return np.where(y_pred >= 0, 1, 0)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
