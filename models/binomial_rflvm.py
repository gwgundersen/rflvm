"""============================================================================
RFLVM with Binomial observations.
============================================================================"""

import autograd.numpy as np
from   autograd.scipy.special import expit as ag_logistic
from   autograd.scipy.special import gammaln as ag_gammaln
from   models._base_logistic_rflvm import _BaseLogisticRFLVM
from   scipy.special import expit as logistic


# -----------------------------------------------------------------------------

class BinomialRFLVM(_BaseLogisticRFLVM):

    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, missing, exposure, disp_prior=10., bias_var=10.):
        """Initialize Binomial RFLVM.
        """
        super().__init__(
            rng=rng,
            data=data,
            n_burn=n_burn,
            n_iters=n_iters,
            latent_dim=latent_dim,
            n_clusters=n_clusters,
            n_rffs=n_rffs,
            dp_prior_obs=dp_prior_obs,
            dp_df=dp_df,
            disp_prior=disp_prior,
            bias_var=bias_var
        )
        self.Y_missing = missing.flatten()
        self.trials = exposure

# -----------------------------------------------------------------------------
# Public API.
# -----------------------------------------------------------------------------

    def predict(self, X, return_latent=False):
        """Predict data `Y` given latent variable `X`.
        """
        phi_X = self.phi(X, self.W, add_bias=True)
        F     = phi_X @ self.beta.T
        Y     = logistic(F)
        if return_latent:
            K = phi_X @ phi_X.T
            return Y * self.trials, F, K
        return Y * self.trials


    def get_params(self):
        """Return model parameters.
        """
        X = self.X_samples if self.t >= self.n_burn else self.X
        return dict(
            X=X,
            W=self.W,
            beta=self.beta
        )

# -----------------------------------------------------------------------------
# Sampling.
# -----------------------------------------------------------------------------

    def log_likelihood(self, **kwargs):
        """Generalized, differentiable log likelihood function.
        """
        # This function can be called for two reasons:
        #
        #   1. Optimize the log likelihood w.r.t. `X`.
        #   2. Evaluate the log likelihood w.r.t. a MH-proposed `W`.
        #
        X = kwargs.get('X', self.X)
        W = kwargs.get('W', self.W)

        phi_X = self.phi(X, W, add_bias=True)
        psi   = phi_X @ self.beta.T
        LL    = self._log_c_func() \
                + (self._a_func() * psi).flatten()[~self.Y_missing] \
                - (self._b_func() * np.log(1 + np.exp(psi))).flatten()[~self.Y_missing]

        return LL.sum()

    def _sample_likelihood_params(self):
        """Sample likelihood- or observation-specific model parameters.
        """
        self._sample_omega()
        self._sample_beta()

    def _evaluate_proposal(self, W_prop):
        """Evaluate Metropolis-Hastings proposal `W_prop`.
        """
        return self.log_likelihood(self.X, W_prop, self.beta)

    def _a_func(self, j=None):
        """See parent class.
        """
        if j is not None:
            return self.Y[:, j]
        return self.Y

    def _b_func(self, j=None):
        """See parent class.
        """
        if j is not None:
            return self.trials[:, j]
        return self.trials

    def _log_c_func(self):
        """See parent class.
        """
        k = self.Y.flatten()[~self.Y_missing]
        n = self.trials.flatten()[~self.Y_missing]
        return ag_gammaln(n+1) - ag_gammaln(k+1) - ag_gammaln(n-k+1)

    def _j_func(self):
        """See parent class.
        """
        return self.J

    def _log_posterior_x(self, X):
        """Compute log posterior of `X`.
        """
        LL = self.log_likelihood(X, self.W, self.beta)
        LP = self._log_prior_x(X)
        return LL + LP

# -----------------------------------------------------------------------------
# Initialization.
# -----------------------------------------------------------------------------

    def _init_specific_params(self):
        """Initialize likelihood-specific parameters.
        """
        pass
