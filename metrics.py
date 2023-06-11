"""============================================================================
Model performance and comparison metrics.
============================================================================"""

import numpy as np
from   sklearn.metrics import (accuracy_score,
                               mean_squared_error as sklearn_mse,
                               r2_score)
from   sklearn.model_selection import KFold
from   sklearn.neighbors import KNeighborsClassifier
from scipy.linalg import svd 
from scipy.spatial import KDTree
from typing import Tuple
from arviz import summary, convert_to_dataset, convert_to_inference_data


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


def get_summary(parameter: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        parameter (np.ndarray): array with (chains, num samples, parameter size ** )
        chains >= 2
    Returns:
        float: returns the summary ndarray
    """
    return summary(convert_to_dataset(parameter)).to_numpy()

def get_posterior_mean(parameter: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        parameter (np.ndarray): array with (chains, num samples, parameter size ** )
        chains >= 2

    Returns:
        np.ndarray: posterior mean
    """
    return convert_to_inference_data(parameter).posterior.mean(dim=["chain", "draw"]).x.to_numpy()



def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):

    p,k = Phi.shape
    R = np.eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u,s,vh = svd(np.dot(Phi.T,np.asarray(Lambda)**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T,Lambda))))))
        R = np.dot(u,vh)
        d = np.sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return np.dot(Phi, R)


def rotate_factors(player_factor_tensor:np.ndarray, use_varimax:bool = True)->Tuple[np.ndarray, np.ndarray]:
    """

    Args:
        player_factor_tensor (np.ndarray): sample x num factors x num players 
        varimax (bool): whether to apply varimax rotation or not

    Returns:
        np.ndarray: sample x num factors x num players  rotated tensor
    """

    n_samples, n_factors, _ = player_factor_tensor.shape
    output_tensor = np.zeros_like(player_factor_tensor)
    rotations = [np.eye(n_factors)]
    output_tensor[0,:,:] = player_factor_tensor[0,:,:] if not use_varimax else varimax(player_factor_tensor[0,:,:])
    for i in range(1,n_samples):
        U, _, V =  svd(output_tensor[0,:,:].dot(player_factor_tensor[i,:,:].T), full_matrices=False)
        rotation = U.dot(V)
        rotations.append(rotation)
        output_tensor[i,:,:] = rotation.dot(player_factor_tensor[i,:,:])
    return output_tensor, np.stack(rotations,axis = 0)


def get_neighbors(data_frame, player, num_k, latent_space, query = ""):
    name_map = data_frame.query(query)[["id","name"]].drop_duplicates().reset_index()[["id","name"]]
    player_index = name_map.index[name_map["name"] == player][0]
    point = latent_space[player_index,:]
    _, indices = KDTree(latent_space).query(point, k = num_k)
    nearest_neighbors_names = name_map.loc[indices]["name"]
    return nearest_neighbors_names