from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import warnings
# We will suppress ConvergenceWarnings in this task. In practice, you should take warnings more seriously.
warnings.filterwarnings("ignore")


def reduce_dimension(X_train: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """
    :param X_train: Training data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality, which has shape (n_samples, n_components), and the PCA object
    """

    # Done: Create a PCA object and fit it using X_train
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_)}")
    return X_train_pca, pca


def train_nn(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier with different number of neurons in one hidden layer.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # DONE: Train MLPClassifier with different number of neurons in one hidden layer.
    #       Print the train accuracy, validation accuracy, and the training loss for each configuration.
    #       Return the MLPClassifier that you consider to be the best.
    n_hidden_values = [2, 10, 100, 200, 500]
    best_classifier = None
    best_val_accuracy = 0

    for n_hidden in n_hidden_values:
        print(f"\n----- Training with {n_hidden} hidden neurons -----")
        mlp = MLPClassifier(hidden_layer_sizes=(n_hidden,), max_iter=500, solver='adam', random_state=1)
        mlp.fit(X_train, y_train)
        
        train_accuracy = mlp.score(X_train, y_train)
        val_accuracy = mlp.score(X_val, y_val)
        
        print(f"Hidden neurons: {n_hidden}")
        print(f"Training accuracy: {train_accuracy}")
        print(f"Validation accuracy: {val_accuracy}")
        print(f"Final training loss: {mlp.loss_}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_classifier = mlp

    return best_classifier


def train_nn_with_regularization(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier using regularization.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Use the code from the `train_nn` function, but add regularization to the MLPClassifier.
    #       Again, return the MLPClassifier that you consider to be the best.
    # Define the configurations to test
    configurations = [
        {"alpha": 0.1, "early_stopping": False},
        {"alpha": 0.0001, "early_stopping": True},  # alpha set to default value
        {"alpha": 0.1, "early_stopping": True}
    ]

    best_classifier = None
    best_val_accuracy = 0
    best_config = None

    for config in configurations:
        print(f"\n----- Training with alpha={config['alpha']} and early_stopping={config['early_stopping']} -----")
        for n_hidden in [2, 10, 100, 200, 500]:
            print(f"\n----- Training with {n_hidden} hidden neurons -----")
            mlp = MLPClassifier(hidden_layer_sizes=(n_hidden,), max_iter=500, solver='adam', random_state=1,
                                alpha=config['alpha'], early_stopping=config['early_stopping'])
            mlp.fit(X_train, y_train)
            
            train_accuracy = mlp.score(X_train, y_train)
            val_accuracy = mlp.score(X_val, y_val)
            
            print(f"Hidden neurons: {n_hidden}")
            print(f"Training accuracy: {train_accuracy}")
            print(f"Validation accuracy: {val_accuracy}")
            print(f"Final training loss: {mlp.loss_}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_classifier = mlp
                best_config = config

    print(f"\nBest configuration: alpha={best_config['alpha']}, early_stopping={best_config['early_stopping']}")
    return best_classifier

    


def plot_training_loss_curve(nn: MLPClassifier) -> None:
    """
    Plot the training loss curve.

    :param nn: The trained MLPClassifier
    """
    # Done: Plot the training loss curve of the MLPClassifier. Don't forget to label the axes.
    plt.plot(nn.loss_curve_)
    plt.title('Training Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()


def show_confusion_matrix_and_classification_report(nn: MLPClassifier, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Plot confusion matrix and print classification report.

    :param nn: The trained MLPClassifier you want to evaluate
    :param X_test: Test features (PCA-projected)
    :param y_test: Test targets
    """
    # TODO: Use `nn` to compute predictions on `X_test`.
    #       Use `confusion_matrix` and `ConfusionMatrixDisplay` to plot the confusion matrix on the test data.
    #       Use `classification_report` to print the classification report.

    y_pred = nn.predict(X_test)

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    # Plot the confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Print the classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    

def perform_grid_search(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Perform GridSearch using GridSearchCV.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The best estimator (MLPClassifier) found by GridSearchCV
    """
    # DONE: Create parameter dictionary for GridSearchCV, as specified in the assignment sheet.
    #       Create an MLPClassifier with the specified default values.
    #       Run the grid search with `cv=5` and (optionally) `verbose=4`.
    #       Print the best score (mean cross validation score) and the best parameter set.
    #       Return the best estimator found by GridSearchCV.
    param_grid = {
        'alpha': [0, 0.1, 1.0],
        'solver': ['lbfgs', 'adam'],
        'hidden_layer_sizes': [(100,), (200,)]
    }

    # Create an MLPClassifier with specified default values
    mlp = MLPClassifier(max_iter=100, random_state=42)

    # Create GridSearchCV with cv=5 and verbose=4
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, verbose=4)

    # Fit GridSearchCV to the data
    grid_search.fit(X_train, y_train)

    # Print the best score and the best parameters
    print(f"Best score: {grid_search.best_score_}")
    print(f"Best parameters: {grid_search.best_params_}")

    # Return the best estimator
    return grid_search.best_estimator_

    return None