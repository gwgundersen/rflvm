"""============================================================================
Utility functions for common visualizations.
============================================================================"""

import numpy as np
import random
import matplotlib.pyplot as plt
from   metrics import affine_align
from   scipy.special import expit as logistic
from scipy.cluster.hierarchy import linkage, leaves_list
from adjustText import adjust_text
# -----------------------------------------------------------------------------
# Base visualizer for data with 2D latent variables.
# -----------------------------------------------------------------------------

class Visualizer:

    def __init__(self, directory, dataset):
        self.directory  = directory
        self.dataset    = dataset
        
        self.x_colors = 'r'
        self.model_name = 'RFLVM'
        if dataset.has_true_X:
            self.plot_X(X=dataset.X, suffix='true')

    def plot_X_init(self, X_init):
        self.plot_X(X=X_init, suffix='init')

    def plot_iteration(self, t, Y, F, K, X, labels = []):
        self.plot_X(t=t, X=X)
        self.plot_K(K = K, t = t, labels = labels)
        if F is not None:
            self.plot_F(t, F)
        if self.dataset.has_true_K and K is not None:
            self.compare_K(t, K)
        self.compare_Y(t, Y)

    def plot_K(self, K, suffix = "", t = -1, labels = []):
        dissimilarity = np.around(1 - np.abs(K), decimals = 10)
        hierarchy = linkage(dissimilarity, method='complete')
        order = leaves_list(hierarchy)
        reordered_matrix = dissimilarity[:, order]
        reordered_matrix = reordered_matrix[order, :]
        plt.imshow(reordered_matrix, cmap='coolwarm_r')
        plt.colorbar()
        if suffix:
            suffix = f'_{suffix}'
        fname = f'{t}_K{suffix}.png'
        plt.title("K Matrix")
        plt.yticks(np.arange(0,len(order),50),labels[order][::50], fontsize=7)
        plt.tight_layout()
        self._save(fname)


    def plot_X(self, X, suffix='', t=-1, labels=[], frac = 1):
        if suffix:
            suffix = f'_{suffix}'
        fname = f'{t}_X{suffix}.png'
        X_aligned = X
        n_samples = int(frac * X_aligned.shape[0])
        random_indices = random.sample(range(X.shape[0]), n_samples)
        plt.scatter(X_aligned[random_indices, 0], X_aligned[random_indices, 1], c=self.x_colors)
        texts = [plt.text(X_aligned[i,0], X_aligned[i,1], labels[i], ha='center', va='center') for i in random_indices]
        # Adjust the text positions to avoid overlap
        adjust_text(texts)

        self._save(fname)

        if self.dataset.has_true_X and suffix not in ['true', 'init']:
            self.compare_X_marginals(X=X, t=t)

    def compare_X_marginals(self, X, suffix='', t=-1):
        fname = f'{t}_X{suffix}_marg.png'
        N, D = X.shape
        fig, axes = plt.subplots(2, 1)
        first = True
        titles = ['x coordinate', 'y coordinate']
        X = affine_align(X, self.dataset.X)
        for ax, x_true, x_est, title in zip(axes, self.dataset.X.T, X.T,
                                            titles):
            ax.plot(range(N), x_true, label='true X', color='blue')
            ax.plot(range(N), x_est, label=self.model_name, color='red')
            if first:
                first = False
                ax.legend()
        self._save(fname)

    def plot_F(self, t, F):
        if self.dataset.has_true_F:
            self._compare_F_or_P(self.dataset.F, F, f'{t}_F.png')
        else:
            fname = f'{t}_F.png'
            self._plot_F_or_P(F, fname)

    def plot_P(self, t, F):
        P = logistic(F)
        if self.dataset.has_true_F:
            P_true = logistic(self.dataset.F)
            self._compare_F_or_P(P_true, P, f'{t}_P.png')
        else:
            fname = f'{t}_P.png'
            self._plot_F_or_P(P, fname)

    def _plot_F_or_P(self, F_or_P, fname):
        fig, axes = plt.subplots(5, 1)
        for ax, f_or_p in zip(axes, F_or_P.T[:5]):
            ax.plot(f_or_p)
        self._save(fname)

    def _compare_F_or_P(self, F_or_P_true, F_or_P, fname):
        fig, axes = plt.subplots(5, 1)
        first = True
        for ax, true, inf in zip(axes, F_or_P_true.T[:5], F_or_P.T[:5]):
            ax.plot(true, label='true')
            ax.plot(inf,  label='learned')
            if first:
                first = False
                ax.legend()
        self._save(fname)

    def compare_K(self, t, K, suffix=''):
        if suffix:
            suffix = f'_{suffix}'
        fname = f'{t}_K{suffix}.png'
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(self.dataset.K)
        ax1.set_title('K true')
        ax2.imshow(K)
        ax2.set_title('K approx')
        self._save(fname)

    def compare_Y(self, t, Y, suffix=''):
        if suffix:
            suffix = f'_{suffix}'
        fname = f'{t}_Y{suffix}.png'
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(self.dataset.Y)
        ax1.set_title('Y true')
        ax2.imshow(Y)
        ax2.set_title(self.model_name)
        self._save(fname)

    def plot_LL(self, LL_list: list[float], model_name = ""):
        """ plots log likelihood line plot

        Args:
            LL_list (list): list of float
        """
        plt.plot(LL_list)
        plt.ylabel("Log Likelihood")
        plt.xlabel("Iteration (factor of 10)")
        plt.title("Log Likelihood vs. MCMC Iteration")
        self._save(f"{model_name}_LL.png")

    def _save(self, fname):
        plt.tight_layout()
        plt.savefig(f'{self.directory}/{fname}', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close('all')
