"""============================================================================
Multivariate t-distribution. Architecture based on SciPy's
`_multivariate.py` module by Joris Vankerschaver 2013.
============================================================================"""

import numpy as np
from   scipy._lib._util import check_random_state
from   scipy.stats._multivariate import _PSD
from   scipy.special import gammaln


# -----------------------------------------------------------------------------

class MultivariateTGenerator:

    def __init__(self, seed=None):
        """Initialize a multivariate t-distributed random variable.

        seed : Random state.
        """
        self._random_state = check_random_state(seed)

    def __call__(self, mean=None, shape=1, df=1, seed=None):
        """Create a frozen multivariate t-distribution. See
        `MultivariateTFrozen` for parameters.
        """
        return MultivariateTFrozen(mean=mean, shape=shape, df=df, seed=seed)

    def logpdf(self, x, mean=None, shape=1, df=1):
        """Log of the multivariate t-distribution probability density function.

        Parameters
        ----------
        x : p-dimensional ndarray or n-by-p matrix.

        mean : p-dimensional ndarray.

        shape : p-by-p positive definite shape matrix. This is not the
                distribution's covariance matrix.

        df : Degrees of freedom.

        Returns
        -------

        logpdf : Log of the probability density function evaluated at `x`.
        """
        mean, shape, df = self._process_parameters(mean, shape, df)
        shape_info = _PSD(shape)
        return self._logpdf(x, mean, shape_info.U, shape_info.log_pdet, df)

    def _logpdf(self, x, mean, U, log_pdet, df):
        """Utility method to bypass processing parameters for frozen
        distributions. See `logpdf` for parameters.
        """
        n_dims    = len(x.shape)
        is_matrix = n_dims == 2
        p         = x.shape[1] if is_matrix else x.size

        dev        = x - mean
        maha       = np.square(np.dot(dev, U)).sum(axis=-1)

        t = 0.5 * (df + p)
        A = gammaln(t)
        B = gammaln(0.5 * df)
        C = p/2. * np.log(df * np.pi)
        D = 0.5 * log_pdet
        E = -t * np.log(1 + (1./df) * maha)

        return A - B - C - D + E

    def pdf(self, x, mean=None, shape=1, df=1):
        """Multivariate t-distribution probability density function. See
        `logpdf` for parameters and return value.
        """
        return np.exp(self.logpdf(x, mean, shape, df))

    def rvs(self, mean=None, shape=1, df=1, size=1, random_state=None):
        """Draw random samples from a multivariate t-distribution.
        """
        mean, shape, df = self._process_parameters(mean, shape, df)
        if random_state is not None:
            rng = check_random_state(random_state)
        else:
            rng = self._random_state

        p = len(mean)
        if df == np.inf:
            x = 1.
        else:
            x = rng.chisquare(df, size=size) / df

        z = rng.multivariate_normal(np.zeros(p), shape, size=size)
        samples = mean + z / np.sqrt(x)[:, None]
        return samples

    def _process_parameters(self, mean, shape, df):
        """Infer dimensionality from mean array and shape matrix, handle
        defaults, and ensure compatible dimensions.
        """
        # Process shape matrix.
        if not np.isscalar(shape) and len(shape.shape) != 2:
            raise ValueError('`shape` matrix must be a scalar or a 2D array.')
        p = 1 if np.isscalar(shape) else shape.shape[0]
        shape = np.asarray(shape, dtype=float)
        if p > 1 and shape.shape[0] != shape.shape[1]:
            raise ValueError('`shape` must be square.')

        # Process mean.
        if mean is not None:
            mean = np.asarray(mean, dtype=float)
        else:
            mean = np.zeros(p)

        # Ensure mean and shape compatible.
        if mean.shape[0] != shape.shape[0]:
            raise ValueError("Arrays `mean` and `shape` ")

        # Process degrees of freedom.
        df = np.asarray(df, dtype=float)
        if df.size != 1:
            msg = 'Degrees of freedom parameter `df` must be scalar.'
            raise ValueError(msg)

        return mean, shape, df


# -----------------------------------------------------------------------------

class MultivariateTFrozen:

    def __init__(self, mean=None, shape=1, df=1, seed=None):
        """Create a frozen multivariate t-distribution.
        See `MultivariateTGenerator` for parameters.
        """
        self._dist = MultivariateTGenerator(seed)
        mean, shape, df = self._dist._process_parameters(mean, shape, df)
        self.shape_info = _PSD(shape)
        self.mean, self.shape, self.df = mean, shape, df

    def logpdf(self, x):
        U = self.shape_info.U
        log_pdet = self.shape_info.log_pdet
        return self._dist._logpdf(x, self.mean, U, log_pdet, self.df)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(mean=self.mean,
                              shape=self.shape,
                              df=self.df,
                              size=size,
                              random_state=random_state)


# -----------------------------------------------------------------------------

multivariate_t = MultivariateTGenerator()
