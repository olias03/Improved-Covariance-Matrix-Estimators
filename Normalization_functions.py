# This Code implements the instantaneous power normalization proposed in
# EEG Signal Processing in MI-BCI Applications with Improved Covariance Matrix Estimators.
# This code has been developed by Javier Olias.
import numpy as np
from scipy.linalg import sqrtm
from numpy import linalg as la


def normalize_covariance(data, normalization_matrix):
    """
    Function that compute the normalized covariance of trials
    :param data: Trial samples. Number of samples X Number of sensors
    :param normalization_matrix: Normalization matrix. Because of computational cost reasons:
    It should be the square root of the inverse of the averaged covariance of the trials.
    :return: Normalized covariance, Normalized data
    """

    standard_samples = data - data.mean(axis=0)
    gamma = np.array([np.mean(np.dot(standard_samples, normalization_matrix) ** 2, axis=1) ** 0.5]).T
    normalized_data = standard_samples / gamma
    aux_cov = np.dot(normalized_data.T, normalized_data)
    return aux_cov / standard_samples.shape[1] ** 2, normalized_data


def normalization_cov(data, max_iter=3, tol=0):
    """
    Compute mean covariance of a set of training trials using the  preprocessing described in Table II
    of "EEG Signal Processing in MI-BCI Applications with Improved Covariance Matrix Estimators"
    :param data: Set of trials. Numpy array (number of trials X number of samples X number of sensors
    :param max_iter: Maximum number of iterations. With 3 iterations is enough
    :param tol: Tolerance to finish before the max_iter is reached
    :return: reference covariance, and the matrix that should be used to normalize the trials.
    """

    data = list(data)
    mean_cov = np.stack(list(map(lambda x1: np.cov(x1.T), data))).mean(axis=0)

    for k in range(max_iter):
        last_mean_cov = mean_cov
        _last_mean_cov = np.linalg.inv(sqrtm(last_mean_cov))
        mean_cov = np.stack(list(map(lambda x1: normalize_covariance(x1, _last_mean_cov)[0], data))).mean(axis=0)
        if np.abs(mean_cov - last_mean_cov).sum() < tol:
            break
    return mean_cov, np.linalg.inv(sqrtm(mean_cov))


def scale_invar_Riemannian_distance(cov1, cov2):
    """
    Evaluates the scale-invariant Riemannian distance
    :param cov1: first  covariance matrix
    :param cov2: second covariance matrix
    :return: scale-invariant Riemannian distance between cov1 and cov2
    """

    eig_values, eig_vectors = la.eig(np.dot(la.inv(cov1), cov2))
    lamb = eig_values * np.exp(-np.mean(np.log(eig_values)))

    return np.sqrt(np.sum((np.log(lamb))**2))
