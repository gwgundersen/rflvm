import autograd.numpy as np
from   autograd import jacobian
from   scipy.optimize import minimize
from models._base_rflvm import _BaseRFLVM
from   scipy.special import expit as logistic
from   autograd.scipy.special import expit as ag_logistic, gammaln as ag_gammaln
from   autograd.scipy.stats import norm as ag_norm, poisson as ag_poisson, multivariate_normal as ag_mvn
from   scipy.linalg import solve_triangular
from   scipy.linalg.lapack import dpotrs
from   pypolyagamma import PyPolyaGamma

class MixedOutputRFLVM(_BaseRFLVM):
    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, missing, exposure, gaussian_indices = None,
                 poisson_indices = None, binomial_indices = None, disp_prior=10., bias_var=1.):
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
            self.exposure_binomial = exposure[:, binomial_indices]
        if poisson_indices is not None:
            self.Y_poisson_missing = missing[:, poisson_indices].flatten()
            self.exposure_poisson = exposure[:, poisson_indices]

        ### logistic params
        
        self.disp_prior = disp_prior
        self.bias_var = bias_var
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
        F = self.F_samples if self.t >= self.n_burn else None
        K = self.K_samples if self.t >= self.n_burn else None
        Beta = self.beta_samples if self.t >= self.n_burn else None
        return dict(
            X=X,
            W=self.W,
            F=F,
            K=K,
            beta=Beta
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
            C = np.sqrt(np.repeat(self.sigma_beta[self.gaussian_indices][None, :], self.N, axis=0))/self.exposure_gaussian
            LL += ag_norm.logpdf(self.Y[:, self.gaussian_indices].flatten()[~self.Y_gaussian_missing],
                                F[:, self.gaussian_indices].flatten()[~self.Y_gaussian_missing],
                                C.flatten()[~self.Y_gaussian_missing]).sum()
        
        if self.poisson_indices is not None:
            theta = np.exp(F[:, self.poisson_indices] + self.exposure_poisson)
            LL += ag_poisson.logpmf(self.Y[:, self.poisson_indices].flatten()[~self.Y_poisson_missing], theta.flatten()[~self.Y_poisson_missing]).sum()
        
        if self.binomial_indices is not None:
            LL  += (self._log_c_func() + self._a_func().flatten()[~self.Y_binomial_missing] + F[:, self.binomial_indices].flatten()[~self.Y_binomial_missing] - self._b_func().flatten()[~self.Y_binomial_missing] * np.log(1 + np.exp(F[:, self.binomial_indices].flatten()[~self.Y_binomial_missing]))).sum()
        
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
        self.sigma_beta = np.ones(self.J) ### variance for each of the beta's 
        self.gamma_a_beta = np.ones(self.J) ### inverse gamma priors
        self.gamma_b_beta = np.ones(self.J) ### inverse gamma priors
        
    

        ### logistic model stuff
        self.pg             = PyPolyaGamma()
        prior_Sigma         = np.eye(self.M+1)
        prior_Sigma[-1, -1] = np.sqrt(self.bias_var)
        self.inv_B          = np.linalg.inv(prior_Sigma)
        mu_A_b              = np.zeros(self.M+1)
        self.inv_B_b        = self.inv_B @ mu_A_b
        self.omega = np.empty(self.Y[:, self.binomial_indices].shape)

    def _sample_mixed_output_beta(self):
        if self.gaussian_indices:
            self._sample_beta_gaussian()
        if self.poisson_indices:
            self._sample_beta_poisson()
        if self.binomial_indices:
            self._sample_omega()
            self._sample_beta_binomial()

    def _sample_beta_gaussian(self):
        """Gibbs sample `beta` and noise parameter `sigma_beta`.
        """
        I = np.eye(self.M + 1)
        phi_X = self.phi(self.X, self.W, add_bias=True)
        a_post = self.gamma_a0 + .5 * self.N
        R = np.diag(np.power(self.exposure_gaussian[:,0],2)) ### same 
        Q = -phi_X.T.dot(R)
        T = Q.dot(Q.T) + I
        B_n = self.Y[:,self.gaussian_indices].T.dot(R - (Q.T).dot(np.linalg.solve(T, Q))).dot(self.Y[:,self.gaussian_indices]) 
        phi_x_lambda_y = -Q.dot(self.Y[:,self.gaussian_indices])

        for index in self.gaussian_indices:
            self.beta[index] = self._sample_gaussian(J = T / self.sigma_beta[index], h = phi_x_lambda_y.T[index] / self.sigma_beta[index])

        # sample from inverse gamma
        
        

        b_post = self.gamma_b0 + .5 * np.diagonal(B_n)

        self.sigma_beta[self.gaussian_indices] = 1. / self.rng.gamma(a_post, 1./b_post)

    
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
            LP = 0
            for i in range(J_poiss):
                LP   += ag_mvn.logpdf(beta[i], self.b0, self.sigma_beta[self.poisson_indices][i]*self.B0).sum()  ### add prior to beta
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

        #### now we update the variances 

        a_post = self.gamma_a_beta[self.poisson_indices] + .5 * (self.M + 1) 
        b_post = self.gamma_b_beta[self.poisson_indices]  + (np.diagonal(beta_map.dot(beta_map.T))) * .5

        self.sigma_beta[self.poisson_indices] = 1/self.rng.gamma(a_post, 1/b_post)



    
    def _sample_beta_binomial(self):
        """Sample `β|ω ~ N(m, V)`. See (Polson 2013).
        """
        phi_X = self.phi(self.X, self.W, add_bias=True)

        for i,j in enumerate(self.binomial_indices):
            # This really computes: phi_X.T @ np.diag(omega[:, j]) @ phi_X
            J = (phi_X * self.omega[:, i][:, None]).T @ phi_X + \
                (self.inv_B * self.sigma_beta[j])
            h = phi_X.T @ self._kappa_func(i) + (self.inv_B_b * self.sigma_beta[j])
            joint_sample = self._sample_gaussian(J=J, h=h)
            self.beta[j] = joint_sample

        ### update the variances 
        a_post = self.gamma_a_beta[self.binomial_indices] + .5 * (self.M + 1) 
        b_post = self.gamma_b_beta[self.binomial_indices]  + np.diagonal(self.beta[self.binomial_indices,:].dot(self.beta[self.binomial_indices,:].T)) * .5

        self.sigma_beta[self.binomial_indices] = 1/self.rng.gamma(a_post, 1/b_post)

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
            return self.exposure_binomial[:, j]
        return self.exposure_binomial

    def _log_c_func(self):
        """See parent class.
        """
        k = self.Y[:, self.binomial_indices].flatten()[~self.Y_binomial_missing]
        n = self.exposure_binomial.flatten()[~self.Y_binomial_missing]
        return ag_gammaln(n+1) - ag_gammaln(k+1) - ag_gammaln(n-k+1)

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
