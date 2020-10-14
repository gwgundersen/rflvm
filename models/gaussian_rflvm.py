"""============================================================================
RFLVM with Gaussian observations.
============================================================================"""

from   autograd import jacobian
import autograd.numpy as np
from   autograd.scipy.special import gammaln
from   autograd.scipy.stats import (norm as ag_norm,
                                    multivariate_normal as ag_mvn)
from   models._base_rflvm import _BaseRFLVM
from   scipy.optimize import minimize


# -----------------------------------------------------------------------------

class GaussianRFLVM(_BaseRFLVM):

    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, marginalize):
        """Initialize Gaussian RFLVM.
        """
        self.marginalize = marginalize

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

# -----------------------------------------------------------------------------
# Public API.
# -----------------------------------------------------------------------------

    def predict(self, X, return_latent=False):
        """Predict data `Y` given latent variable `X`.
        """
        phi_X = self.phi(X, self.W)

        if self.marginalize:
            # This is the mean of the posterior predictive in Bayesian linear
            # regression, a multivariate t-distribution.
            Lambda_n = phi_X.T @ phi_X + np.eye(self.M)
            mu_n     = np.linalg.inv(Lambda_n) @ phi_X.T @ self.Y
            Y = F    = phi_X @ mu_n
        else:
            phi_X = self.phi(X, self.W, add_bias=True)
            Y = F = phi_X @ self.beta.T

        if return_latent:
            K = phi_X @ phi_X.T
            return Y, F, K
        return Y

    def log_likelihood(self, **kwargs):
        """Differentiable log likelihood.
        """
        X = kwargs.get('X', self.X)
        W = kwargs.get('W', self.W)

        if self.marginalize:
            return self.log_marginal_likelihood(X, W)
        else:
            beta = kwargs.get('beta', self.beta)
            phi_X = self.phi(X, W, add_bias=True)
            F = phi_X @ beta.T
            LL = ag_norm.logpdf(self.Y.flatten(), F.flatten(), 1).sum()
            return LL

    def log_marginal_likelihood(self, X, W):
        """Log marginal likelihood after integrating out `beta`. We assume
        the prior mean of `beta` is zero and that `S_0 = identity(M)`.
        """
        phi_X = self.phi(X, W)
        S_n   = phi_X.T @ phi_X + np.eye(self.M)
        mu_n  = np.linalg.inv(S_n) @ phi_X.T @ self.Y
        a_n   = self.a0 + self.N / 2
        A     = np.diag(self.Y.T @ self.Y)
        C     = np.diag(mu_n.T @ S_n @ mu_n)
        b_n   = self.b0 + 0.5 * (A - C)

        # Compute Lambda term.
        sign, logdet = np.linalg.slogdet(S_n)
        lambda_term  = -0.5 * sign * logdet

        # Compute b_n term.
        b_term = self.a0 * np.log(self.b0) - a_n * np.log(b_n)

        # Compute a_n term.
        gamma_term = gammaln(a_n) - gammaln(self.a0)

        # Compute sum over all y_n.
        return np.sum(gamma_term + b_term + lambda_term)

    def get_params(self):
        """Return model parameters.
        """
        X = self.X_samples if self.t >= self.n_burn else self.X
        return dict(
            X=X,
            W=self.W
        )

# -----------------------------------------------------------------------------
# Sampling.
# -----------------------------------------------------------------------------

    def _sample_likelihood_params(self):
        """Sample likelihood- or observation-specific model parameters.
        """
        if self.marginalize:
            # We integrated out `beta` a la Bayesian linear regression.
            pass
        else:
            self._sample_beta()

    def _sample_beta(self):
        """Compute the maximum a posteriori estimation of `beta`.
        """
        def _neg_log_posterior(beta_flat):
            beta = beta_flat.reshape(self.J, self.M + 1)
            LL = self.log_likelihood(beta=beta)
            LP = ag_mvn.logpdf(beta, self.b0, self.B0).sum()
            return -(LL + LP)

        resp = minimize(_neg_log_posterior,
                        x0=np.copy(self.beta),
                        jac=jacobian(_neg_log_posterior),
                        method='L-BFGS-B',
                        options=dict(
                            maxiter=self.max_iters
                        ))
        beta_map = resp.x.reshape(self.J, self.M + 1)
        self.beta = beta_map

    def _evaluate_proposal(self, W_prop):
        """Evaluate Metropolis-Hastings proposal `W` using the log evidence.
        """
        if self.marginalize:
            return self.log_marginal_likelihood(self.X, W_prop)
        else:
            return self.log_likelihood(W=W_prop)

    def _log_posterior_x(self, X):
        """Compute log posterior of `X`.
        """
        if self.marginalize:
            LL = self.log_marginal_likelihood(X, self.W)
        else:
            LL = self.log_likelihood(X=X)
        LP = self._log_prior_x(X)
        return LL + LP

# -----------------------------------------------------------------------------
# Initialization.
# -----------------------------------------------------------------------------

    def _init_specific_params(self):
        """Initialize likelihood-specific parameters.
        """
        if self.marginalize:
            # Equivalent to the inverse-gamma hyperparameters in Bayesian
            # linear regression.
            self.a0 = 1
            self.b0 = 1
        else:
            # Linear coefficients β in `Poisson(exp(phi(X)*β))`.
            self.b0 = np.zeros(self.M + 1)
            self.B0 = np.eye(self.M + 1)
            self.beta = self.rng.multivariate_normal(self.b0, self.B0,
                                                     size=self.J)
