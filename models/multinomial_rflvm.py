"""============================================================================
RFLVM with multinomial observations.

In-comment citations:

    (Baker 1994)   The multinomial-Poisson transformation
    (Chen 2013)    Scalable inference for logistic-normal topic models
    (Polson 2013)  Bayesian inference for logistic models using Polya-Gamma
                   latent variables
============================================================================"""


import autograd.numpy as np
from   autograd import jacobian
from   autograd.scipy.special import logsumexp as ag_lse
from   autograd.scipy.stats import poisson as ag_poisson
from   autograd.scipy.stats import norm as ag_norm
from   models._base_logistic_rflvm import _BaseLogisticRFLVM
from   scipy.optimize import minimize


# -----------------------------------------------------------------------------

class MultinomialRFLVM(_BaseLogisticRFLVM):

    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, disp_prior=10., bias_var=10.,
                 A_var=100.):
        """Initialize RFLVM.
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

        # `Q` is an `(N, J)` matrix in which each component is
        # `Q_n - \sum x_{nj}`.
        self.Q_n = self.Y.sum(axis=1)
        self.Q = self.Q_n[:, None] * np.ones(self.Y.shape)

        # Fix last beta to zero.
        self.beta = np.vstack((self.beta,np.zeros(self.M+1)))
    
        # Prior variance of normalizing constant.
        self.A_var = A_var

# -----------------------------------------------------------------------------
# Public API.
# -------------------------------s---------------------------------------------

    def predict(self, X, return_latent=False):
        """Predict data `Y` given latent variable `X`.
        """
        phi_X = self.phi(X, self.W, add_bias=True)
        psi   = phi_X @ self.beta.T
        pi    = self.psi_to_pi(psi)
        Y     = self.Q_n[:, None] * pi
        if return_latent:
            F = psi
            K = phi_X @ phi_X.T
            return Y, F, K
        return Y

    def psi_to_pi(self, psi):
        """Log-normalize and exponentiate psi vector        
        """
        return np.exp(psi - ag_lse(psi, axis=1)[:, None])

    def log_likelihood(self, **kwargs):
        """Differentiable log likelihood for the multinomial distribution.

        We have to overwrite `_BaseLogisticRFLVM`'s log likelihood function
        because this model's log likelihood is multinomial, which is not
        in the "logistic family".
        """
        # Optimize the log likelihood w.r.t. `X`. Use MH to evaluate the log
        # likelihood w.r.t. a proposed `W`.
        X = kwargs.get('X', self.X)
        W = kwargs.get('W', self.W)

        phi_X = self.phi(X, W, add_bias=True)
        psi   = phi_X @ self.beta.T + self.a0[:, None]

        return ag_poisson.logpmf(self.Y, np.exp(psi)).sum()

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

    def _sample_likelihood_params(self):
        """Sample likelihood- or observation-specific model parameters.
        """
        self._sample_omega()
        self._sample_beta()
        self._sample_a()

    def _sample_a(self):
        """Optimize the nuisance parameter in the multinomial-Poisson
        transformation. See (Baker 1994) for more details.
        """
        def _neg_log_posterior(a0):
            phi_X = self.phi(self.X, self.W, add_bias=True)
            psi   = phi_X @ self.beta.T 
            # Assume prior mean on a0 is the actual normalizing constant.
            prior_mean = -ag_lse(psi, axis=1)
            psi += a0[:, None]
            LL   = ag_poisson.logpmf(self.Y, np.exp(psi)).sum()
            var  = np.sqrt(self.A_var)*np.ones(self.N)
            LL  += ag_norm.logpdf(a0, prior_mean, var).sum()
            return(-1.*LL)

        resp = minimize(_neg_log_posterior,
                        x0=np.copy(self.a0),
                        jac=jacobian(_neg_log_posterior),
                        method='L-BFGS-B',
                        options=dict(
                            maxiter=self.max_iters
                        ))
        self.a0 = resp.x.reshape(self.N)

    def _sample_beta(self):
        """Sample β|ω ~ N(m, V). See (Polson 2013).
        """
        phi_X = self.phi(self.X, self.W, add_bias=True)
        for j in range(self._j_func()):
            notj = np.arange(self.J) != j
            ksi = ag_lse(phi_X @ self.beta[notj].T, axis=1)
            # This really computes: phi_X.T @ np.diag(omega[:, j]) @ phi_X
            J = (phi_X * self.omega[:, j][:, None]).T @ phi_X + \
                self.inv_B
            h = phi_X.T @ (self._kappa_func(j) + ksi*self.omega[:, j]) + \
                self.inv_B_b 
            self.beta[j] = self._sample_gaussian(J=J, h=h)

    def _sample_omega(self):
        """Sample ω|β ~ PG(y+r, x*β). See (Polson 2013).
        """
        phi_X = self.phi(self.X, self.W, add_bias=True)
        # Is there a faster way to do the below line? It's inefficient.
        Ksi = np.vstack([
            ag_lse(phi_X @ self.beta[np.arange(self.J) != j].T, axis=1)
            for j in range(self.J)
        ]).T
        psi = (phi_X @ self.beta.T) - Ksi
        bb = self._b_func()
        self.pg.pgdrawv(bb.ravel(),
                        psi.ravel(),
                        self.omega.ravel())
        self.omega = self.omega.reshape(self.Y.shape)

    def _evaluate_proposal(self, W_prop):
        """Evaluate Metropolis-Hastings proposal `W_prop`.
        """
        return self.log_likelihood(W=W_prop)

    def _j_func(self):
        """See parent class.
        """
        return self.J-1

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
            return self.Q[:,j]
        return self.Q

    def _log_posterior_x(self, X):
        """Compute log posterior of `X`.
        """
        LL = self.log_likelihood(X=X)
        LP = self._log_prior_x(X)
        return LL + LP

# -----------------------------------------------------------------------------
# Initialization.
# -----------------------------------------------------------------------------

    def _init_specific_params(self):
        """Initialize likelihood-specific parameters.
        """
        # Initialize nuisance parameters for multinomial-Poisson transform
        self.a0 = np.zeros(self.N)
