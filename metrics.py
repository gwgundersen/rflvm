"""============================================================================
Model performance and comparison metrics.
============================================================================"""

import numpy as np
from   sklearn.metrics import (accuracy_score,
                               mean_squared_error as sklearn_mse,
                               r2_score)
from   sklearn.model_selection import KFold
from   sklearn.neighbors import KNeighborsClassifier
from arviz import ess, rhat, convert_to_dataset


# -----------------------------------------------------------------------------

def r_squared(X, X_true):
    """Return R^2 error *after* best affine transformation of `X` onto
    `X_true`.
    """
    X_aligned = affine_align(X, X_true)
    return r2_score(X_true, X_aligned)


# -----------------------------------------------------------------------------

def mean_squared_error(val1, val2):
    """Return mean squared error. Wraps Scikit-learn so that we can (1) change
    implementations if we want and (2) import metrics from a single module.
    """
    return sklearn_mse(val1, val2)


# -----------------------------------------------------------------------------

def knn_classify(X, y, rng, n_splits=5):
    """Run K-nearest neighbors algorithm using K-fold validation. `X` is the
    inferred latent variable and `y` are the labels, e.g. digits for MNIST and
    political parties for Congress 109.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=rng)
    accs = np.empty(n_splits)
    for i, (train_index, test_index) in enumerate(kf.split(X)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        m = KNeighborsClassifier(n_neighbors=1)
        m.fit(X=X_train, y=y_train)
        y_pred = m.predict(X_test)
        accs[i] = accuracy_score(y_test, y_pred)

    return accs.mean()


# -----------------------------------------------------------------------------

def affine_align(X, X_true=None, return_residuals=False):
    """Since the inferred `X` is unidentifiable, we find the best affine
    transformation.
    """
    if X_true is None:
        return X
    # The bias allows for arbitrary translation.
    ones = np.ones(len(X))[:, None]
    X = np.hstack([ones, X])
    # Least squares finds best linear map.
    W, res, _, _ = np.linalg.lstsq(a=X, b=X_true, rcond=None)
    if return_residuals:
        return X @ W, res.sum()
    return X @ W


def get_rhat(parameter: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        parameter (np.ndarray): array with (chains, num samples, parameter size ** )
        chains >= 2
    Returns:
        float: returns the rhat ndarray
    """
    return rhat(convert_to_dataset(parameter)).x.to_numpy()

def get_ess(parameter: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        parameter (np.ndarray): 

    Returns:
        float: return ndarray of effective sample size
    """

    return ess(convert_to_dataset(parameter)).x.to_numpy()
