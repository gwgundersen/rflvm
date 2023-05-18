import autograd.numpy as np
from   autograd import jacobian
from   scipy.optimize import minimize
from models._base_rflvm import _BaseRFLVM
from   scipy.special import expit as logistic
from   autograd.scipy.special import expit as ag_logistic
from   autograd.scipy.stats import norm as ag_norm, poisson as ag_poisson, multivariate_normal as ag_mvn
from   scipy.linalg import solve_triangular
from   scipy.linalg.lapack import dpotrs

class MixedOutputRFLVM(_BaseRFLVM):
    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, missing, exposure, gaussian_indices = None,
                 poisson_indices = None, binomial_indices = None):
        """Initialize Mixed Output RFLVM.
        """

        self.marginalize = False
        self.poisson_indices = poisson_indices
        self.gaussian_indices = gaussian_indices
        self.binomial_indices = binomial_indices
        self.Y_gaussian_missing = None
        self.Y_poisson_missing = None
        self.Y_binomial_missing = None
        self.exposure_gaussian = None
        self.exposure_poisson = None
        self.exposure_binomial = None
        if gaussian_indices is not None:
            self.Y_gaussian_missing = missing[:,gaussian_indices].flatten()
            self.exposure_gaussian = exposure[:, gaussian_indices]
        if binomial_indices is not None:
            self.Y_binomial_missing = missing[:, binomial_indices].flatten()
            self.exposure_binomial = missing[:, binomial_indices]
        if poisson_indices is not None:
            self.Y_poisson_missing = missing[:, poisson_indices].flatten()
            self.exposure_poisson = missing[:, poisson_indices]

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

        
    
    def predict(self, X, return_latent=False):
        """Predict data `Y` given latent variable `X`.
        """
        phi_X = self.phi(X, self.W, add_bias=True)
        F     = phi_X @ self.beta.T
        Y     = F 
        if self.binomial_indices is not None:
            Y[:, self.binomial_indices] = logistic(F[:, self.binomial_indices]) * self.exposure_binomial
        if self.gaussian_indices is not None:
            pass
        if self.poisson_indices is not None:
            Y[:, self.poisson_indices] = np.exp(F[:, self.poisson_indices] + self.exposure_poisson)
        if return_latent:
            K = phi_X @ phi_X.T
            return Y , F, K
        return Y
    
    def get_params(self):
        """Return model parameters.
        """
        X = self.X_samples if self.t >= self.n_burn else self.X
        return dict(
            X=X,
            W=self.W
        )

    def log_likelihood(self, **kwargs):
        """Differentiable log likelihood.
        """
        X = kwargs.get('X', self.X)
        W = kwargs.get('W', self.W)


        beta = kwargs.get('beta', self.beta)
        phi_X = self.phi(X, W, add_bias=True)
        F = phi_X @ beta.T 
        # Explicitly shape before flattening to ensure elements align.
        LL = 0
        if self.gaussian_indices is not None:
            C = np.sqrt(np.repeat(self.sigma_y[None, :], self.N, axis=0))/self.exposure_gaussian
            LL += ag_norm.logpdf(self.Y[:, self.gaussian_indices].flatten()[~self.Y_gaussian_missing],
                                F[:, self.gaussian_indices].flatten()[~self.Y_gaussian_missing],
                                C.flatten()[~self.Y_gaussian_missing]).sum()
        
        if self.poisson_indices is not None:
            theta = np.exp(F[:, self.poisson_indices] + self.exposure_poisson)
            LL += ag_poisson.logpmf(self.Y[:, self.poisson_indices].flatten()[~self.Y_poisson_missing], theta.flatten()[~self.Y_poisson_missing]).sum()
        
        if self.binomial_indices is not None:
            theta = ag_logistic(F[:, self.binomial_indices])
            k = self.Y[:, self.binomial_indices].flatten()[~self.Y_binomial_missing]
            n = self.exposure_binomial.flatten()[~self.Y_binomial_missing]
            p = theta.flatten()[~self.Y_binomial_missing]
            LL  += (np.log(p)*(k) + (n-k)*np.log(1-p)).sum()
        
        return LL
    
    def _init_specific_params(self):
        """Initialize likelihood-specific parameters.
        """
        # Equivalent to the inverse-gamma hyperparameters in Bayesian
        # linear regression.
        self.gamma_a0 = 1
        self.gamma_b0 = 1
        # Linear coefficients β in `Poisson(exp(phi(X)*β))`.
        self.b0   = np.zeros(self.M + 1)
        self.B0   = np.eye(self.M + 1)
        self.beta = self.rng.multivariate_normal(self.b0, self.B0,
                                                    size=self.J)
        self.sigma_y = np.ones(len(self.gaussian_indices)) if self.gaussian_indices is not None else None 
        self.omega = np.empty(self.Y[:, self.binomial_indices].shape)

    def _sample_mixed_output_beta(self):
        if self.gaussian_indices is not None:
            self._sample_beta_gaussian
        if self.poisson_indices is not None:
            self._sample_beta_poisson
        if self.binomial_indices is not None:
            self._sample_beta_binomial

    def _sample_beta_gaussian(self):
        """Gibbs sample `beta` and noise parameter `sigma_y`.
        """
        J_gauss = len(self.gaussian_indices)
        phi_X = self.phi(self.X, self.W, add_bias=True)
        cov_j = self.B0 + phi_X.T @ phi_X
        mu_j  = np.tile((self.B0 @ self.b0), (J_gauss, 1)).T + \
                (phi_X.T @ self.Y[:,self.gaussian_indices])
        # multi-output generalization of mvn sample code
        L  = np.linalg.cholesky(cov_j)
        Z  = self.rng.normal(size=self.beta[self.gaussian_indices,:].shape).T
        LZ = solve_triangular(L, Z, lower=True, trans='T')
        L_mu = dpotrs(L, mu_j, lower=True)[0]
        self.beta[self.gaussian_indices, :] = (LZ + L_mu).T
        # sample from inverse gamma
        a_post = self.gamma_a0 + .5 * self.N
        b_post = self.gamma_b0 + .5 * np.diag(
            (self.Y[:, self.gaussian_indices].T @ self.Y[:, self.gaussian_indices]) + \
            (self.b0 @ self.B0 @ self.b0.T) + \
            (mu_j.T @ np.linalg.solve(cov_j, mu_j))
        )
        self.sigma_y = 1. / self.rng.gamma(a_post, 1./b_post)
    
    def _sample_beta_poisson(self):
        """Compute the maximum a posteriori estimation of `beta`.
        """
        J_poiss = len(self.poisson_indices)
        def _neg_log_posterior(beta_flat):
            beta = beta_flat.reshape(J_poiss, self.M+1)
            phi_X = self.phi(self.X, self.W, add_bias=True)
            F = phi_X @ beta.T 
            theta = np.exp(F + self.exposure_poisson)
            LL = ag_poisson.logpmf(self.Y[:, self.poisson_indices].flatten()[~self.Y_poisson_missing],
                                    theta.flatten()[~self.Y_poisson_missing]).sum()
            LP   = ag_mvn.logpdf(beta, self.b0, self.B0).sum()
            return -(LL + LP)

        resp = minimize(_neg_log_posterior,
                        x0=np.copy(self.beta[self.poisson_indices,:]),
                        jac=jacobian(_neg_log_posterior),
                        method='L-BFGS-B',
                        options=dict(
                            maxiter=self.max_iters
                        ))
        beta_map = resp.x.reshape(J_poiss, self.M+1)
        self.beta[self.poisson_indices,:] = beta_map
    
    def _sample_beta_binomial(self):
        """Sample `β|ω ~ N(m, V)`. See (Polson 2013).
        """
        phi_X = self.phi(self.X, self.W, add_bias=True)

        for i,j in enumerate(self.binomial_indices):
            # This really computes: phi_X.T @ np.diag(omega[:, j]) @ phi_X
            J = (phi_X * self.omega[:, i][:, None]).T @ phi_X + \
                self.inv_B
            h = phi_X.T @ self._kappa_func(i) + self.inv_B_b
            joint_sample = self._sample_gaussian(J=J, h=h)
            self.beta[j] = joint_sample
    
    def _a_func(self, j=None):
        """See parent class.
        """
        if j is not None:
            return self.Y[:, self.binomial_indices][:, j]
        return self.Y[:, self.binomial_indices]

    def _b_func(self, j=None):
        """See parent class.
        """
        if j is not None:
            return np.ones(self.Y[:, self.binomial_indices][:, j].shape)
        return np.ones(self.Y[:, self.binomial_indices].shape)

    def _log_c_func(self):
        """See parent class.
        """
        return 0

    def _j_func(self):
        """See parent class.
        """
        return len(self.binomial_indices)

    def _kappa_func(self, j):
        """This function returns `kappa(y)`. See the comment at the top of this
        file and (Polson 2013).
        """
        return self._a_func(j) - (self._b_func(j) / 2.0)
    
    def _sample_omega(self):
        """Sample `ω|β ~ PG(b, x*β)`. See (Polson 2013).
        """
        phi_X = self.phi(self.X, self.W, add_bias=True)
        psi   = phi_X @ self.beta[self.binomial_indices,:].T
        b     = self._b_func()
        self.pg.pgdrawv(b.ravel(),
                        psi.ravel(),
                        self.omega.ravel())
        self.omega = self.omega.reshape(self.Y[:, self.binomial_indices].shape)


    def _sample_likelihood_params(self):
        """Sample likelihood- or observation-specific model parameters.
        """
        self._sample_mixed_output_beta()
    


    def _evaluate_proposal(self, W_prop):
        """Evaluate Metropolis-Hastings proposal `W` using the log evidence.
        """
        
        return self.log_likelihood(W=W_prop)

    def _log_posterior_x(self, X):
        """Compute log posterior of `X`.
        """
        
        LL = self.log_likelihood(X=X)
        LP = self._log_prior_x(X)
        return LL + LP
