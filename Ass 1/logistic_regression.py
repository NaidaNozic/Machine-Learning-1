import numpy as np

def create_design_matrix_dataset_1(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 1.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # Created design matrix X for dataset 1
    feature1 = np.zeros((X_data.shape[0], 1))
    X = np.hstack((X_data, feature1))

    for i in range(X.shape[0]):
        x1 = X[i, 0]
        x2 = X[i, 1]
        if x1 >= 10 and x1 <= 29 and x2 >= 0 and x2 <= 20:
            X[i, 2] = 1

    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def create_design_matrix_dataset_2(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 2.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # Created the design matrix X for dataset 2
    feature1 = np.zeros((X_data.shape[0], 1))
    X = np.hstack((X_data, feature1))
    for i in range(X.shape[0]):
        x1 = X[i, 0]
        x2 = X[i, 1]
        X[i, 2] = x1**2 + x2**2 - 24*24

    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def create_design_matrix_dataset_3(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 3.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # Create the design matrix X for dataset 3
    feature1 = np.zeros((X_data.shape[0], 1))
    feature2 = np.zeros((X_data.shape[0], 1))
    feature3 = np.zeros((X_data.shape[0], 1))
    feature4 = np.zeros((X_data.shape[0], 1))
    X = np.hstack((X_data, feature1))
    X = np.hstack((X, feature2))
    X = np.hstack((X, feature3))
    X = np.hstack((X, feature4))
    for i in range(X.shape[0]):
        x1 = X[i, 0]
        X[i, 2] = np.cos(2.5*x1)
        X[i, 4] = x1**3/4
        X[i, 5] = x1**2/3
        if x1 < 0:
            X[i, 3] = 1

    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def logistic_regression_params_sklearn():
    """
    :return: Return a dictionary with the parameters to be used in the LogisticRegression model from sklearn.
    Read the docs at https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    # TODO: Try different `penalty` parameters for the LogisticRegression model
    return {'penalty': 'l2'}
