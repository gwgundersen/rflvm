"""============================================================================
Abstract dataset attributes.
============================================================================"""


class Dataset:

    def __init__(self, name, is_categorical, Y, X=None, F=None, K=None, R=None,
                 labels=None, latent_dim=None, missing = None, exposure = None):

        self.name = name
        self.R = R

        if is_categorical and labels is None:
            raise ValueError('Labels must be provided for categorical data.')
        self.is_categorical = is_categorical
        self.has_missing = missing is not None
        self.has_true_X = X is not None
        self.has_true_F = F is not None
        self.has_true_K = K is not None
        self.has_labels = labels is not None

        self._latent_dim = latent_dim

        self.Y = Y
        self.F = F
        self.K = K
        self.X = X
        self.R = R
        self.labels = labels
        self.exposure = exposure
        assert Y.shape == missing.shape ## make sure missing indicator 
        self.Y_missing = missing

    def __str__(self):
        return f"<class 'datasets.Dataset ({self.name})'>"

    def __repr__(self):
        return str(self)

    @property
    def latent_dim(self):
        if self._latent_dim:
            return self._latent_dim
        elif self.has_true_X:
            return self.X.shape[1]
        else:
            return 2
