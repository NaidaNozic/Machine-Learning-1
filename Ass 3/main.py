import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from utils.plotting import plot_dataset, plot_decision_boundary
from utils.datasets import get_toy_dataset
from knn import KNearestNeighborsClassifier
from svm import LinearSVM
from sklearn.ensemble import RandomForestClassifier


def grid_search_knn_and_plot_decision_boundary(X_train, y_train, X_test, y_test, dataset_name):
    knn = KNearestNeighborsClassifier()
    # Done: Use the `GridSearchCV` meta-classifier and search over different values of `k`
    #       Include the `return_train_score=True` option to get the training accuraciesls

    # Specify the parameter grid to explore various `k` values
    parameter_grid = {'k': range(1, 101)}

    # Initialize GridSearchCV to determine the optimal `k`
    grid_search = GridSearchCV(knn, parameter_grid, cv=5, return_train_score=True)
    grid_search.fit(X_train, y_train)
    

    # this plots the decision boundary
    plt.figure()
    plot_decision_boundary(X_train, grid_search)
    plot_dataset(X_train, y_train, X_test, y_test)
    plt.title(f"Decision boundary for dataset {dataset_name}\nwith k={grid_search.best_params_['k']}")
    # Done you should use the plt.savefig(...) function to store your plots before calling plt.show()
    plt.savefig(f"decision_boundary_{dataset_name}.png")
    plt.show()

    # Done: Create a plot that shows the mean training and validation scores (y axis)
    #       for each k \in {1,...,100} (x axis).
    #       Hint: Check the `cv_results_` attribute of the `GridSearchCV` object
    # Extract mean training and validation scores from the GridSearchCV results
    mean_train_scores = grid_search.cv_results_['mean_train_score']
    mean_test_scores = grid_search.cv_results_['mean_test_score']
    k_values = range(1, 101)

    # Plot mean training and validation scores for each value of `k`
    plt.figure()
    plt.plot(k_values, mean_train_scores, label='Mean Training Score')
    plt.plot(k_values, mean_test_scores, label='Mean Validation Score')
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.title(f'Mean Training and Validation Scores for Different k Values\nDataset: {dataset_name}')
    plt.legend()
    plt.savefig(f"training_validation_scores_{dataset_name}.png")
    plt.show()
    # Manually select the best k based on the highest mean validation accuracy
    
    optimal_k = k_values[np.argmax(mean_test_scores)]
    optimal_knn = KNearestNeighborsClassifier(k=optimal_k)
    optimal_knn.fit(X_train, y_train)

    # Evaluate the classifier performance on the test set with the selected k
    test_accuracy = optimal_knn.score(X_test, y_test)
    print(f"Optimal k for {dataset_name} dataset (manual selection): {optimal_k}")
    print(f"Test set accuracy for {dataset_name} dataset with k={optimal_k}: {test_accuracy * 100:.2f}%")


def task1_2():
    print('-' * 10, 'Task 1.2', '-' * 10)
    for idx in [1, 2, 3]:
        X_train, X_test, y_train, y_test = get_toy_dataset(idx)
        grid_search_knn_and_plot_decision_boundary(X_train, y_train, X_test, y_test, dataset_name=idx)


def task1_4():
    print('-' * 10, 'Task 1.4', '-' * 10)
    dataset_name = '2 (noisy)'
    X_train, X_test, y_train, y_test = get_toy_dataset(2, apply_noise=True)
    for k in [1, 30, 100]:
        # Done: Fit your KNearestNeighborsClassifier with k in {1, 30, 100} and plot the decision boundaries.
        #       You can use the `cross_val_score` method to manually perform cross-validation.
        #       Report the mean cross-validated scores.
        knn = KNearestNeighborsClassifier(k=k)
        knn.fit(X_train, y_train)

        # Perform cross-validation and report the mean cross-validated scores
        cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
        print(f"Mean cross-validated score for k={k}: {np.mean(cv_scores):.2f}")

        # Plot decision boundaries without the test set
        plt.figure()
        plt.title(f"Decision Boundary for Dataset {dataset_name}\nwith k={k}")
        plot_decision_boundary(X_train, knn)
        plot_dataset(X_train, y_train)
        plt.show()

    # Find the best parameters for the noisy dataset using grid search
    grid_search_knn_and_plot_decision_boundary(X_train, y_train, X_test, y_test, dataset_name=dataset_name)


def task2_2():
    print('-' * 10, 'Task 2.2', '-' * 10)

    X_train, X_test, y_train, y_test = get_toy_dataset(1, remove_outlier=True)
    svm = LinearSVM()
    '''param_grid = {
        'C': 10. ** np.arange(-3, 4),
        'eta': 10. ** np.arange(-5, -2)
    }'''
    param_grid = {
        'C': 50. ** np.arange(-3, 4),
        'eta': 10. ** np.arange(-5, -2)
    }
    '''param_grid = {
        'C': 60. ** np.arange(-8, 8),
        'eta': 10. ** np.arange(-5, -2)
    }'''
    # Use grid search to find suitable parameters.
    grid_search = GridSearchCV(svm, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print("best parameters:",grid_search.best_params_)
    print("cross validated score:",grid_search.best_score_)

    # Use the parameters you have found to instantiate a LinearSVM.
    # The `fit` method returns a list of scores that you should plot in order to monitor the convergence.
    best_params = grid_search.best_params_
    svm = LinearSVM(C=best_params['C'], eta=best_params['eta'])
    loss_list = svm.fit(X_train, y_train)

    # Plot the loss over iterations
    plt.figure()
    plt.plot(loss_list)
    plt.title('Loss over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    # This plots the decision boundary
    plt.figure()
    plot_dataset(X_train, y_train, X_test, y_test)
    plot_decision_boundary(X_train, svm)
    plt.show()


def task2_3():
    print('-' * 10, 'Task 2.3', '-' * 10)
    for idx in [1, 2, 3]:

        X_train, X_test, y_train, y_test = get_toy_dataset(idx)
        # Perform grid search, decide on suitable parameter ranges
        # and state sensible parameter ranges in your report
        param_grid = {'C': [0.1, 1, 10],
                      'kernel': ['linear', 'rbf'],
                      'gamma': [0.1, 1, 10]}
        svc = SVC(tol=1e-4)
        grid_search = GridSearchCV(svc, param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Using the best parameter settings, report the score on the test dataset (X_test, y_test)
        print(f"Best parameters - {grid_search.best_params_}, Mean CV accuracy - {grid_search.best_score_}")

        test_accuracy = grid_search.score(X_test, y_test)
        print(f"Accuracy on test set: {test_accuracy}")

        # This plots the decision boundary
        plt.figure()
        plt.title(f"[SVM] Decision boundary for dataset {idx}\nwith params={grid_search.best_params_}")
        plot_dataset(X_train, y_train, X_test, y_test)
        plot_decision_boundary(X_train, grid_search)
        plt.show()


def task3_1():
    n_estimators_list = [1, 100]
    max_depth_list = np.arange(1, 26)
    for idx in [1, 2, 3]:
        X_train, X_test, y_train, y_test = get_toy_dataset(idx)
        cv_val_accuracy = {}
        cv_train_accuracy = {}
        for n_estimators in n_estimators_list:
            # TODO: Instantiate a RandomForestClassifier with n_estimators and random_state=0
            #       and use GridSearchCV over max_depth_list to find the best max_depth.
            #       Use `return_train_score=True` to get the training accuracies during CV.
            grid_search = None

            # TODO: Store `mean_test_score` and `mean_train_score` in cv_val_accuracy and cv_train_accuracy.
            #       The dictionary key should be the number of estimators.
            #       Hint: Check the `cv_results_` attribute of the `GridSearchCV` object
            cv_val_accuracy[n_estimators] = None
            cv_train_accuracy[n_estimators] = None

            # This plots the decision boundary with just the training dataset
            plt.figure()
            plot_decision_boundary(X_train, grid_search)
            plot_dataset(X_train, y_train)
            plt.title(f"Decision boundary for dataset {idx}\n"
                      f"n_estimators={n_estimators}, max_depth={grid_search.best_params_['max_depth']}")
            plt.show()

        # TODO: Create a plot that shows the mean training and validation scores (y axis)
        #       for each max_depth in max_depth_list (x axis).
        #       Use different colors for each n_estimators and linestyle="--" for validation scores.

    # TODO: Instantiate a RandomForestClassifier with the best parameters for each dataset and
    #       report the test scores (using X_test, y_test) for each dataset.

def task3_bonus():
    X_train, X_test, y_train, y_test = get_toy_dataset(4)

    # TODO: Find suitable parameters for an SVC and fit it.
    #       Report mean CV accuracy of the model you choose.

    # TODO: Fit a RandomForestClassifier with appropriate parameters.

    # TODO: Create a `barh` plot of the `feature_importances_` of the RF classifier.
    #       See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html

    # TODO: Use recursive feature elimination to automatically choose the best number of parameters.
    #       Set `scoring='accuracy'` to look for the feature subset with highest accuracy and fit the RFECV
    #       to the training dataset. You can pass the classifier from the previous step to RFECV.
    #       See https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html

    # TODO: Use the RFECV to transform the training dataset -- it automatically removes the least important
    #       feature columns from the datasets. You don't have to change y_train or y_test.
    #       Fit an SVC classifier with appropriate parameters on the new dataset and report the mean CV score.
    #       Do you see a difference in performance when compared to the previous dataset? Report your findings.

    # TODO: If the CV performance of this SVC is better, transform the test dataset as well and report the test score.
    #       If the performance is worse, report the test score of the previous SVC.


if __name__ == '__main__':
    # Task 1.1 consists of implementing the KNearestNeighborsClassifier class
    task1_2()
    # Task 1.3 does not need code to be answered
    task1_4()

    # Task 2.1 consists of a pen & paper exercise and the implementation of the LinearSVM class
    task2_2()
    task2_3()

    task3_1()
    # Task 3.2 is a theory question
    task3_bonus()