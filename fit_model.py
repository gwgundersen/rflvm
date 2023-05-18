"""============================================================================
Fit random feature latent variable model.
============================================================================"""

import argparse
from   datasets import load_dataset
from   logger import (format_number,
                      Logger)
from   models import (BernoulliRFLVM,
                      BinomialRFLVM,
                      GaussianRFLVM,
                      MultinomialRFLVM,
                      MixedOutputRFLVM,
                      NegativeBinomialRFLVM,
                      PoissonRFLVM)
from   metrics import (knn_classify,
                       mean_squared_error,
                       r_squared)
import numpy as np
from   numpy.random import RandomState
from   pathlib import Path
import pickle
from   time import perf_counter
from   visualizer import Visualizer


# -----------------------------------------------------------------------------

def fit_log_plot(args):
    """Fit model, plot visualizations, log metrics.
    """
    # Configure logging, dataset, and visualizer.
    # -------------------------------------------
    p = Path(args.directory)
    if not p.exists():
        p.mkdir()
    log = Logger(directory=args.directory)
    log.log(f'Initializing RNG with seed {args.seed}.')
    rng = RandomState(args.seed)
    ds  = load_dataset(rng, args.dataset, args.emissions, metric_list = args.metric.split(",")[0] if len(args.metric.split(",")) <= 1 else args.metric.split(","), model = args.model, 
                       exposure_list = args.exposure.split(",")[0] if len(args.exposure.split(",")) <= 1 else args.exposure.split(","), age = args.age, gaussian_indices = args.gaussian_indices,
                       poisson_indices = args.poisson_indices, binomial_indices = args.binomial_indices)
    viz = Visualizer(args.directory, ds)

    # Set values on `args` so that they are logged.
    args.n_burn       = int(args.n_iters / 2)  # Recommended in Gelman's BDA.
    args.dp_prior_obs = ds.latent_dim
    args.dp_df        = ds.latent_dim + 1
    args.marginalize  = bool(args.marginalize)
    args.log_every    = 10

    log.log_hline()
    log.log_args(args)

    # Initialize model.
    # -----------------
    if args.model == 'bernoulli':
        model = BernoulliRFLVM(
            rng=rng,
            data=ds.Y,
            n_burn=args.n_burn,
            n_iters=args.n_iters,
            latent_dim=ds.latent_dim,
            n_clusters=args.n_clusters,
            n_rffs=args.n_rffs,
            dp_prior_obs=args.dp_prior_obs,
            dp_df=args.dp_df
        )
    elif args.model == "binomial":
        model = BinomialRFLVM(
            rng=rng,
            data=ds.Y,
            n_burn=args.n_burn,
            n_iters=args.n_iters,
            latent_dim=ds.latent_dim,
            n_clusters=args.n_clusters,
            n_rffs=args.n_rffs,
            dp_prior_obs=args.dp_prior_obs,
            dp_df=args.dp_df,
            missing = ds.Y_missing,
            exposure=ds.exposure
        )
    elif args.model == 'gaussian':
        model = GaussianRFLVM(
            rng=rng,
            data=ds.Y,
            n_burn=args.n_burn,
            n_iters=args.n_iters,
            latent_dim=ds.latent_dim,
            n_clusters=args.n_clusters,
            n_rffs=args.n_rffs,
            dp_prior_obs=args.dp_prior_obs,
            dp_df=args.dp_df,
            marginalize=args.marginalize,
            missing = ds.Y_missing,
            exposure=ds.exposure
        )
    elif args.model == 'poisson':
        model = PoissonRFLVM(
            rng=rng,
            data=ds.Y,
            n_burn=args.n_burn,
            n_iters=args.n_iters,
            latent_dim=ds.latent_dim,
            n_clusters=args.n_clusters,
            n_rffs=args.n_rffs,
            dp_prior_obs=args.dp_prior_obs,
            dp_df=args.dp_df,
            missing=ds.Y_missing,
            exposure=ds.exposure
        )
    elif args.model == 'mixed':
        model = MixedOutputRFLVM(
            rng=rng,
            data=ds.Y,
            n_burn=args.n_burn,
            n_iters=args.n_iters,
            latent_dim=ds.latent_dim,
            n_clusters=args.n_clusters,
            n_rffs=args.n_rffs,
            dp_prior_obs=args.dp_prior_obs,
            dp_df=args.dp_df,
            missing=ds.Y_missing,
            exposure=ds.exposure,
            gaussian_indices=args.gaussian_indices,
            poisson_indices=args.poisson_indices,
            binomial_indices=args.binomial_indices
        )
    elif args.model == 'multinomial':
        model = MultinomialRFLVM(
            rng=rng,
            data=ds.Y,
            n_burn=args.n_burn,
            n_iters=args.n_iters,
            latent_dim=ds.latent_dim,
            n_clusters=args.n_clusters,
            n_rffs=args.n_rffs,
            dp_prior_obs=args.dp_prior_obs,
            dp_df=args.dp_df
        )
    elif args.model == 'negbinom':
        if args.dataset == 's-curve' and args.emissions == 'gaussian':
            raise NotImplementedError('Sampling `R` requires `Y` to be count '
                                      'data but emissions are Gaussian.')
        model = NegativeBinomialRFLVM(
            rng=rng,
            data=ds.Y,
            n_burn=args.n_burn,
            n_iters=args.n_iters,
            latent_dim=ds.latent_dim,
            n_clusters=args.n_clusters,
            n_rffs=args.n_rffs,
            dp_prior_obs=args.dp_prior_obs,
            dp_df=args.dp_df
        )

    if args.model != args.emissions and args.dataset == 's-curve':
        model_name = model.__class__.__name__
        log.log_hline()
        log.log(f'WARNING: Model is {model_name}, but emissions '
                f'are {args.emissions}. Was this intended?')

    # Visualize the initial value of `X`.
    viz.plot_X_init(model.X)

    # Fit model.
    # ----------
    s_start = perf_counter()
    for t in range(args.n_iters):
        s = perf_counter()
        model.step()
        e = perf_counter() - s

        if t == model.n_burn:
            log.log_hline()
            log.log(f'Burn in complete on iter = {t}. Now plotting using mean '
                    f'of `X` samples after burn in.')
        if (t % args.log_every == 0) or (t == args.n_iters - 1):
            assert(model.t-1 == t)
            plot_and_print(t, rng, log, viz, ds, model, e)

    elapsed_time = (perf_counter() - s_start) / 3600
    log.log_hline()
    log.log(f'Finished job in {format_number(elapsed_time)} (hrs).')
    log.log_hline()


# -----------------------------------------------------------------------------

def plot_and_print(t, rng, log, viz, ds, model, elapsed_time):
    """Utility function for plotting images and printing logs.
    """
    # Generate model predictions.
    # ---------------------------
    Y_pred, F_pred, K_pred = model.predict(model.X, return_latent=True)

    # Plot visualizations.
    # --------------------
    viz.plot_iteration(t, Y_pred, F_pred, K_pred, model.X)

    log.log_hline()
    log.log(t)

    # Log metrics.
    # ------------
    mse_Y = mean_squared_error(Y_pred, ds.Y)
    log.log_pair('MSE Y', mse_Y)

    if ds.has_true_F:
        mse_F = mean_squared_error(F_pred, ds.F)
        log.log_pair('MSE F', mse_F)

    if ds.has_true_K:
        mse_K = mean_squared_error(K_pred, ds.K)
        log.log_pair('MSE K', mse_K)

    if ds.has_true_X:
        r2_X = r_squared(model.X, ds.X)
        log.log_pair('R2 X', r2_X)

    if ds.is_categorical:
        knn_acc = knn_classify(model.X, ds.labels, rng)
        log.log_pair('KNN acc', knn_acc)

    # Log parameters.
    # ---------------
    log.log_pair('DPMM LL', model.calc_dpgmm_ll())
    log.log_pair('K', model.Z_count.tolist())
    log.log_pair('alpha', model.alpha)
    n_mh_iters = (model.t + 1) * model.M
    log.log_pair('W MH acc', model.mh_accept / n_mh_iters)

    if hasattr(model, 'R'):
        log.log_pair('R median', np.median(model.R))

    # Record time.
    # ------------
    log.log_pair('time', elapsed_time)

    # Flush and save state.
    # ---------------------
    params = model.get_params()
    fpath = f'{args.directory}/{args.model}_{args.metric}_rflvm.pickle'
    fpath_model = f'{args.directory}/{args.model}_{args.metric}_model_rflvm.pickle'
    pickle.dump(params, open(fpath, 'wb'))
    pickle.dump(model, open(fpath_model,"wb"))

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    EMISSIONS = ['bernoulli', 'gaussian', 'multinomial', 'negbinom', 'poisson', 'binomial', "mixed"]

    p = argparse.ArgumentParser()
    p.add_argument('--directory',
                   help='Experimental directory.',
                   required=False,
                   default='experiments')
    p.add_argument('--model',
                   help='Model to fit.',
                   required=False,
                   default='gaussian',
                   choices=EMISSIONS)
    p.add_argument('--seed',
                   help='Random seed.',
                   required=False,
                   default=0,
                   type=int)
    p.add_argument('--dataset',
                   help='Experimental dataset.',
                   type=str,
                   default='s-curve',
                   choices=['bridges', 'congress', 's-curve', "bball"])
    p.add_argument('--emissions',
                   help='Emissions used S-curve dataset.',
                   required=False,
                   type=str,
                   default='gaussian',
                   choices=EMISSIONS)
    p.add_argument('--metric',
                   help="metric for bball",
                   required = False,
                   type = str,
                   default = [])
    p.add_argument('--age',
                   help="metric for bball",
                   required = False,
                   type = int,
                   default = 25)
    p.add_argument('--exposure',
                   help="exposure features for bball",
                   required = False,
                   type = str,
                   default = "minutes")
    p.add_argument('--gaussian_indices',
                   help="mixed feature for bball",
                   required = False,
                   type = lambda x: [int(a) for a in x.split(",")],
                   default = [])
    p.add_argument('--poisson_indices',
                   help="mixed feature for bball",
                   required = False,
                   type = lambda x: [int(a) for a in x.split(",")],
                   default = [])
    p.add_argument('--binomial_indices',
                   help="mixed feature for bball",
                   required = False,
                   type = lambda x: [int(a) for a in x.split(",")],
                   default = [])
    p.add_argument('--n_iters',
                   help='Number of iterations for the Gibbs sampler.',
                   required=False,
                   type=int,
                   default=2000)
    p.add_argument('--n_rffs',
                   help='Number of random Fourier features.',
                   required=False,
                   type=int,
                   default=100)
    p.add_argument('--marginalize',
                   help='Whether or not to marginalize out `beta` in the '
                        'Gaussian model.',
                   type=int,
                   default=1)
    p.add_argument('--n_clusters',
                   help='Number of initial clusters for `W`.',
                   required=False,
                   type=int,
                   default=1)

    # Parse and validate script arguments.
    # ------------------------------------
    args = p.parse_args()

    fit_log_plot(args)
