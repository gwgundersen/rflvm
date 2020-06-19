"""============================================================================
Base class for logistic RFLVMs. Logistic models have likelihoods that can be
written as:

    c(y) * [exp(beta * x)^{a(y)} / (1 + exp(beta * x)^{b(y)}]

(Polson 2013) introduce Polya-gamma random variables, which introduces another
function of the data, `kappa(y)`. Sub-classing this model primarily requires to
implementing functions to compute `a(y)`, `b(y)`, `log c(y)`, and `kappa(y)`.

The logic in this class borrows heavily from the Linderman's `PyPolyaGamma`:

    https://github.com/slinderman/pypolyagamma

In-comment citations:

    (Polson 2013)  Bayesian inference for logistic models using Polya-Gamma
                   latent variables
============================================================================"""

import autograd.numpy as np
from   models._base_rflvm import _BaseRFLVM
from   pypolyagamma import PyPolyaGamma


# -----------------------------------------------------------------------------

class _BaseLogisticRFLVM(_BaseRFLVM):

    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, disp_prior, bias_var):
        """Initialize base class for logistic RFLVMs.
        """
        # `_BaseRFLVM` will call `_init_specific_params`, and these need to be
        # set first.
        self.disp_prior = disp_prior
        self.bias_var   = bias_var

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

        # Polya-gamma augmentation.
        self.pg             = PyPolyaGamma()
        prior_Sigma         = np.eye(self.M+1)
        prior_Sigma[-1, -1] = np.sqrt(self.bias_var)
        self.inv_B          = np.linalg.inv(prior_Sigma)
        mu_A_b              = np.zeros(self.M+1)
        self.inv_B_b        = self.inv_B @ mu_A_b
        self.omega          = np.empty(self.Y.shape)

        # Linear coefficients `beta`.
        b0 = np.zeros(self.M+1)
        B0 = np.eye(self.M+1)
        self.beta = self.rng.multivariate_normal(b0, B0, size=self.J)

# -----------------------------------------------------------------------------
# Public API.
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
                + self._a_func() * psi \
                - self._b_func() * np.log(1 + np.exp(psi))

        return LL.sum()

# -----------------------------------------------------------------------------
# Polya-gamma augmentation.
# -----------------------------------------------------------------------------

    def _sample_beta(self):
        """Sample `β|ω ~ N(m, V)`. See (Polson 2013).
        """
        phi_X = self.phi(self.X, self.W, add_bias=True)

        for j in range(self.J):
            # This really computes: phi_X.T @ np.diag(omega[:, j]) @ phi_X
            J = (phi_X * self.omega[:, j][:, None]).T @ phi_X + \
                self.inv_B
            h = phi_X.T @ self._kappa_func(j) + self.inv_B_b
            joint_sample = self._sample_gaussian(J=J, h=h)
            self.beta[j] = joint_sample

    def _sample_omega(self):
        """Sample `ω|β ~ PG(b, x*β)`. See (Polson 2013).
        """
        phi_X = self.phi(self.X, self.W, add_bias=True)
        psi   = phi_X @ self.beta.T
        b     = self._b_func()
        self.pg.pgdrawv(b.ravel(),
                        psi.ravel(),
                        self.omega.ravel())
        self.omega = self.omega.reshape(self.Y.shape)

    def _a_func(self, j=None):
        """This function returns `a(y)`. See the comment at the top of this
        file and (Polson 2013).
        """
        raise NotImplementedError()

    def _b_func(self, j=None):
        """This function returns `b(y)`. See the comment at the top of this
        file and (Polson 2013).
        """
        raise NotImplementedError()

    def _log_c_func(self):
        """This function returns `log c(y)`. This is the normalizer in logistic
        models and is only used in the log likelihood calculation. See the
        comment at the top of this file and (Polson 2013).
        """
        raise NotImplementedError()

    def _kappa_func(self, j):
        """This function returns `kappa(y)`. See the comment at the top of this
        file and (Polson 2013).
        """
        return self._a_func(j) - (self._b_func(j) / 2.0)
