import numpy as np
from models.poisson_rflvm import PoissonRFLVM
from models.binomial_rflvm import BinomialRFLVM
from models.gaussian_rflvm import GaussianRFLVM
from   scipy.special import expit as logistic

class MixedOutputRFLVM(GaussianRFLVM, PoissonRFLVM, BinomialRFLVM):
    def __init__(self, rng, data, n_burn, n_iters, latent_dim, n_clusters,
                 n_rffs, dp_prior_obs, dp_df, missing, exposure, gaussian_indices = None,
                 poisson_indices = None, binomial_indices = None):
        """Initialize Mixed Output RFLVM.
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
    
    
        


