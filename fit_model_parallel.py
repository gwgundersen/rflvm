import argparse
import pandas as pd
import multiprocessing
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
from   metrics import (get_neighbors, get_summary, rotate_factors, get_posterior_mean)
import numpy as np
import pickle
from   numpy.random import RandomState
from   pathlib import Path
from   visualizer import Visualizer

def fit(args):
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

    
    model.fit()
    return model.get_params()

def parallel_fit(args, num_chains):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-2)
    results = pool.map(fit, [args] * num_chains)
    pool.close()
    pool.join()
    return results

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
    p.add_argument('--n_chain',
                   type=int,
                   default=1)

    # Parse and validate script arguments.
    # ------------------------------------

    args = p.parse_args()
    p = Path(args.directory)
    if not p.exists():
        p.mkdir()
    log = Logger(directory=args.directory)
    chains = args.n_chain

    rng = RandomState()
    ds  = load_dataset(rng, args.dataset, args.emissions, metric_list = args.metric.split(",")[0] if len(args.metric.split(",")) <= 1 else args.metric.split(","), model = args.model, 
                        exposure_list = args.exposure.split(",")[0] if len(args.exposure.split(",")) <= 1 else args.exposure.split(","), age = args.age, gaussian_indices = args.gaussian_indices,
                        poisson_indices = args.poisson_indices, binomial_indices = args.binomial_indices)

    # Set values on `args` so that they are logged.
    args.n_burn       = int(args.n_iters / 2)  # Recommended in Gelman's BDA.
    args.dp_prior_obs = ds.latent_dim
    args.dp_df        = ds.latent_dim + 1
    args.marginalize  = bool(args.marginalize)
    args.log_every    = 10

    log.log_hline()
    log.log_args(args)

    results = parallel_fit(args, chains)
    fpath = f'{args.directory}/{args.model}_{args.metric}_{args.n_chain}_rflvm.pickle'
    chained_params = {sample:np.stack([param[sample] for param in results], axis = 0) for sample in results[0].keys()}
    with open(fpath, "wb") as f:
        pickle.dump(chained_params, f)


    