"""============================================================================
Dataset loading functions.
============================================================================"""

from   datasets.dataset import Dataset
from   GPy import kern
import numpy as np
import pandas as pd
from   scipy.special import (expit as logistic,
                             logsumexp)
from   sklearn.datasets import make_s_curve


# -----------------------------------------------------------------------------

def load_dataset(rng, name, emissions, *args):
    """Given a dataset string, returns data and possibly true generative
    parameters.
    """
    loader = {
        'bridges' : load_bridges,
        'congress': load_congress,
        's-curve' : gen_s_curve,
        "bball": load_bball
    }[name]
    if name == 's-curve':
        return loader(rng, emissions)
    else:
        return loader(*args)


# -----------------------------------------------------------------------------

def load_bridges():
    """Load NYC bridges dataset:

    https://data.cityofnewyork.us/Transportation/
      Bicycle-Counts-for-East-River-Bridges/gua4-p9wg
    """
    data   = np.load(f'datasets/bridges.npy', allow_pickle=True)
    data   = data[()]
    Y      = data['Y']
    labels = data['labels']
    return Dataset('bridges', True, Y, labels=labels)


# -----------------------------------------------------------------------------

def load_congress():
    """Congress 109 data:

    https://github.com/jgscott/STA380/blob/master/data/congress109.csv
    https://github.com/jgscott/STA380/blob/master/data/congress109members.csv
    """
    df1 = pd.read_csv(f'datasets/congress109.csv')
    df2 = pd.read_csv(f'datasets/congress109members.csv')
    assert (len(df1) == len(df2))

    # Ensure same ordering.
    df1 = df1.sort_values(by='name')
    df2 = df2.sort_values(by='name')

    Y = df1.values[:, 1:].astype(int)
    labels = np.array([0 if x == 'R' else 1 for x in df2.party.values])
    return Dataset('congress109', True, Y, labels=labels)


def load_bball(metric, model):
    """ Load bball data

    Returns:
        Dataset: dataset of the nba data
    """

    df = pd.read_csv("datasets/player_data.csv")
    df = df.sort_values(by=["id","year"])
    df = df[[metric, "id", "age", "minutes"]]
    df_offset = df[["id", "age", "minutes"]]
    df  = df.pivot(columns="age",values=metric,index="id")
    offset = 0
    if model == "poisson":
        offset = np.log(df_offset.pivot(columns="age", values="minutes",index="id").fillna(1).to_numpy())
    return Dataset("bball", False, Y = df.fillna(df.mean()).to_numpy(), missing = df.isnull().to_numpy(), offset=offset)
    



# -----------------------------------------------------------------------------
# Datasets with synthetic latent spaces.
# -----------------------------------------------------------------------------

def gen_s_curve(rng, emissions):
    """Generate synthetic data from datasets generating process.
    """
    N = 500
    J = 100
    D = 2

    # Generate latent manifold.
    # -------------------------
    X, t = make_s_curve(N, random_state=rng)
    X    = np.delete(X, obj=1, axis=1)
    X    = X / np.std(X, axis=0)
    inds = t.argsort()
    X    = X[inds]
    t    = t[inds]

    # Generate kernel `K` and latent GP-distributed maps `F`.
    # -------------------------------------------------------
    K = kern.RBF(input_dim=D, lengthscale=1).K(X)
    F = rng.multivariate_normal(np.zeros(N), K, size=J).T

    # Generate emissions using `F` and/or `K`.
    # ----------------------------------------
    if emissions == 'bernoulli':
        P = logistic(F)
        Y = rng.binomial(1, P).astype(np.double)
        return Dataset('s-curve', False, Y, X, F, K, None, t)
    if emissions == 'gaussian':
        Y = F + np.random.normal(0, scale=0.5, size=F.shape)
        return Dataset('s-curve', False, Y, X, F, K, None, t)
    elif emissions == 'multinomial':
        C = 100
        pi = np.exp(F - logsumexp(F, axis=1)[:, None])
        Y = np.zeros(pi.shape)
        for n in range(N):
            Y[n] = rng.multinomial(C, pi[n])
        return Dataset('s-curve', False, Y, X, F, K, None, t)
    elif emissions == 'negbinom':
        P = logistic(F)
        R = np.arange(1, J+1, dtype=float)
        Y = rng.negative_binomial(R, 1-P)
        return Dataset('s-curve', False, Y, X, F, K, R, t)
    else:
        assert(emissions == 'poisson')
        theta = np.exp(F)
        Y = rng.poisson(theta)
        return Dataset('s-curve', False, Y, X, F, K, None, t)
