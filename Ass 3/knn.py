import numpy as np
from sklearn.base import BaseEstimator


class KNearestNeighborsClassifier(BaseEstimator):
    def __init__(self, k=1):
        self.k = k
        self.X = None
        self.y = None
        self.classes = None # a list of unique classes in our classification problem

    def fit(self, X, y):
        # Done: Implement this method by storing X, y and infer the unique classes from y
        #       Useful numpy methods: np.unique
        # Save the training dataset
        self.y = y
        self.X = X
        # Determine the distinct classes present in y
        self.classes = np.unique(y)
        return self
        

    def predict(self, X):
        # Done: Predict the class labels for the data on the rows of X
        #       Useful numpy methods: np.argsort, np.argmax
        #       Broadcasting is really useful for this task.
        #       See https://numpy.org/doc/stable/user/basics.broadcasting.html
        # Compute the Euclidean distances between each test sample and all training samples
        distance_matrix = np.sqrt(((self.X - X[:, np.newaxis]) ** 2).sum(axis=2))
        # Identify the indices of the k nearest neighbors
        k_neighbors_indices = np.argsort(distance_matrix, axis=1)[:, :self.k]
        # Extract the labels of the k nearest neighbors
        k_neighbors_labels = self.y[k_neighbors_indices]
        # Predict the class by taking a majority vote among the k nearest labels
        y_pred = np.array([np.argmax(np.bincount(labels)) for labels in k_neighbors_labels])
        return y_pred
        

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
