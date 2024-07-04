from typing import Tuple
import numpy as np
from scipy.stats import multivariate_normal


def calculate_responsibilities(X: np.ndarray, means: np.ndarray, sigmas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    :param X: data for clustering, shape: (N, D), with N being the number of data points, D the dimension
    :param means: means of K D-dimensional Gaussians, shape: (K, D)
    :param sigmas: covariance matrices for K D-dimensional Gaussians, shape: (K, D, D)
    :param weights: component weights (weights for each Gaussian component), an array, shape (K, )
    :return: responsibilities - Equation (5) from the HW4 sheet
    """
    N, K = X.shape[0], means.shape[0]

    # DONE: Calculate responsibilities. You can use the multivariate_normal.pdf function from scipy.stats, see
    #       https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
    #       Note that `multivariate_normal` is already imported.
    responsibilities = np.zeros((N, K))  # Stores all \gamma_{ik} from the HW sheet
    # Calculate the numerator of \gamma_{ik} for each data point i and component k
    for k in range(K):
        # For component k, calculate the weighted probability density for all data points
        responsibilities[:, k] = weights[k] * multivariate_normal.pdf(X, mean=means[k], cov=sigmas[k])
    
    # Calculate the denominator of \gamma_{ik} for each data point i
    # This is the sum of the numerators across all components for each data point
    sum_responsibilities = np.sum(responsibilities, axis=1, keepdims=True)
    
    # Normalize the responsibilities by dividing each numerator by the sum of numerators for the corresponding data point
    responsibilities /= sum_responsibilities
    return responsibilities


def update_parameters(X: np.ndarray, means: np.ndarray, sigmas: np.ndarray,
                      weights: np.ndarray, responsibilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param means: means of K D-dimensional Gaussians, shape: (K, D)
    :param sigmas: covariance matrices for K D-dimensional Gaussians, shape: (K, D, D)
    :param weights: component weights (weights for each Gaussian component), an array, shape (K, )
    :param responsibilities: responsibilities for each data point i and cluster k: shape (N, K)
    :return: means_new - Equation (6) from the HW4 sheet, shape: (K, D),
             sigmas_new - Equation (7) from the HW4 sheet, shape: (K, D, D),
             weights_new - Equation (8) from the HW4 sheet, an array: shape (K, )
    """
    N, K = X.shape[0], means.shape[0]

    # DONE: Calculate means_new, sigmas_new, weights_new using responsibilities
    means_new = np.zeros_like(means)
    sigmas_new = np.zeros_like(sigmas)
    weights_new = np.zeros_like(weights)
    # Calculate means_new   
    for k in range(K):
        means_new[k] = np.sum(responsibilities[:, k][:, np.newaxis] * X, axis=0) / np.sum(responsibilities[:, k])
    # Calculate sigmas_new
    for k in range(K):
        diff = X - means_new[k]
        sigmas_new[k] = np.dot((responsibilities[:, k][:, np.newaxis] * diff).T, diff) / np.sum(responsibilities[:, k])
     # Calculate weights_new
    weights_new = np.sum(responsibilities, axis=0) / N
    return means_new, sigmas_new, weights_new


def em(X: np.ndarray, K: int, max_iter: int, eps=1e-2, init_variance=1.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param max_iter: maximum number of iterations for the EM algorithm.
                     If the algorithm converges earlier, it should also stop earlier.
    :param eps: threshold for the stopping criterion based on the change of the log-likelihood
    :param init_variance: initial variance for the covariance matrices (initialized to be isotropic)
    :return: means - means of K D-dimensional Gaussians, shape: (K, D)
             soft_clusters - soft assignment of data points to clusters, shape: (N, K)
             log_likelihood - an array with values of log-likelihood function over the iterations
    """

    assert max_iter > 0, 'max_iter must be a positive integer'
    N, D = X.shape[0], X.shape[1]

    # Init GMM
    means = np.random.random(size=(K, D))
    cov_mat = np.eye(D) * init_variance
    sigmas = np.repeat(cov_mat[None, :, :], K, axis=0)
    weights = 1. / K * np.ones((K,))
    assert np.isclose(np.sum(weights), 1.0), 'Mixture weights must sum to 1.0'
    print(f'Init: {means=}\n {sigmas=}\n {weights=}')

    log_likelihood = []
    for it in range(max_iter):
        # E-Step
        # DONE: Call the appropriate function
        responsibilities = calculate_responsibilities(X, means, sigmas, weights)
        # M-Step
        # DONE: Call the appropriate function
        means, sigmas, weights = update_parameters(X, means, sigmas, weights, responsibilities)
        # Evaluate log-likelihood under the current model (with parameters means, sigmas, weights)
        soft_clusters = np.zeros((N, K))
        for k in range(K):
            soft_clusters[:, k] = weights[k] * multivariate_normal.pdf(X, mean=means[k, :], cov=sigmas[k])

        log_likelihood.append(np.sum(np.log(np.sum(soft_clusters, axis=1))))

        if it > 1 and np.abs(log_likelihood[-1] - log_likelihood[-2]) < eps:
            print(f'Algorithm converged at iteration {it}.')
            break

    print(f'Fitted parameters: {means=}\n {sigmas=}\n {weights=}')
    return means, soft_clusters, np.array(log_likelihood)
