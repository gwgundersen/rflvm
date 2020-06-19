# Latent Variable Modeling with Random Features

A Python implementation of random feature latent variable models (RFLVMs). See [Latent Variable Modeling with Random Features]() for details.

### Abstract

> Gaussian process-based latent variable models are flexible and theoretically grounded tools for nonlinear dimension reduction, but generalizing to non-Gaussian data likelihoods within this nonlinear framework is statistically challenging. Here, we use random features to develop a family of nonlinear dimension reduction models that are easily extensible to non-Gaussian data likelihoods; we call these _random feature latent variable models_ (RFLVMs). By approximating a nonlinear relationship between the latent space and the observations with a function that is linear with respect to random features, we induce closed-form gradients of the posterior distribution with respect to the latent variable. This allows the RFLVM framework to support computationally tractable nonlinear latent variable models for a variety of data likelihoods in the exponential family without specialized derivations. Our generalized RFLVMs produce results comparable with other state-of-the-art dimension reduction methods on diverse types of data, including neural spike train recordings, images, and text data.

### Demo

To replicate results, call `fit_model.py` with appropriate arguments. The datasets `bridges`, `congress`, and `s-curve` are supported here. For example, to replicate the Gaussian model, call

```python
python fit_model.py --dataset=s-curve --emissions=gaussian --model=gaussian
```

See the `ArgumentParser` instance in `fit_model.py` for a description of arguments.

### Installation

This implementation requires Python 3.X. See `requirements.txt` for a list installed packages and their versions. The main packages are:

```python
autograd
GPy
matplotlib
numpy
pandas
pypolyagamma
scipy
scikit-learn
```
