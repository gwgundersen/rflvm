"""============================================================================
RFLVM with Poisson observations.
============================================================================"""

from   autograd import jacobian
import autograd.numpy as np
from   autograd.scipy.stats import (multivariate_normal as ag_mvn,
                                    poisson as ag_poisson)
from   models._base_rflvm import _BaseRFLVM
from   scipy.optimize import minimize


# -----------------------------------------------------------------------------

class PoissonRFLVM(_BaseRFLVM):

    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, missing, offset):
        """Initialize Poisson RFLVM.
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
            dp_df=dp_df
        )
        self.Y_missing = missing.flatten()
        self.offset = offset

# -----------------------------------------------------------------------------
# Public API.
# -----------------------------------------------------------------------------

    def predict(self, X, return_latent=False):
        """Predict data `Y` given latent variable `X`.
        """
        phi_X = self.phi(X, self.W, add_bias=True)
        F     = phi_X @ self.beta.T
        theta = np.exp(F + self.offset)
        if return_latent:
            K = phi_X @ phi_X.T
            return theta, F, K
        return theta

    def log_likelihood(self, X, W, beta):
        """Differentiable log likelihood.
        """
        phi_X = self.phi(X, W, add_bias=True)
        F     = phi_X @ beta.T
        theta = np.exp(F + self.offset)
        LL    = ag_poisson.logpmf(self.Y.flatten()[~self.Y_missing], theta.flatten()[~self.Y_missing]).sum()
        return LL

    def get_params(self):
        """Return model parameters.
        """
        X = self.X_samples if self.t >= self.n_burn else self.X
        return dict(
            X=X,
            beta=self.beta,
            W=self.W
        )

# -----------------------------------------------------------------------------
# Sampling.
# -----------------------------------------------------------------------------

    def _sample_likelihood_params(self):
        """Sample likelihood- or observation-specific model parameters.
        """
        self._sample_beta()

    def _evaluate_proposal(self, W_prop):
        """Evaluate Metropolis-Hastings proposal `W_prop`.
        """
        return self.log_likelihood(self.X, W_prop, self.beta)

    def _log_posterior_x(self, X):
        """Compute log posterior of `X`.
        """
        LL = self.log_likelihood(X, self.W, self.beta)
        LP = self._log_prior_x(X)
        return LL + LP

    def _sample_beta(self):
        """Compute the maximum a posteriori estimation of `beta`.
        """
        def _neg_log_posterior(beta_flat):
            beta = beta_flat.reshape(self.J, self.M+1)
            LL   = self.log_likelihood(self.X, self.W, beta)
            LP   = ag_mvn.logpdf(beta, self.b0, self.B0).sum()
            return -(LL + LP)

        resp = minimize(_neg_log_posterior,
                        x0=np.copy(self.beta),
                        jac=jacobian(_neg_log_posterior),
                        method='L-BFGS-B',
                        options=dict(
                            maxiter=self.max_iters
                        ))
        beta_map = resp.x.reshape(self.J, self.M+1)
        self.beta = beta_map

# -----------------------------------------------------------------------------
# Initialization.
# -----------------------------------------------------------------------------

    def _init_specific_params(self):
        """Initialize likelihood-specific parameters.
        """
        # Linear coefficients β in `Poisson(exp(phi(X)*β))`.
        self.b0 = np.zeros(self.M + 1)
        self.B0 = np.eye(self.M + 1)
        self.beta  = self.rng.multivariate_normal(self.b0, self.B0,
                                                  size=self.J)
