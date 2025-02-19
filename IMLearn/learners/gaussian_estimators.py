from __future__ import annotations

import math
import numpy as np
from numpy.linalg import inv, det, slogdet

class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.fitted_ = True

        # this mu maximizes the likelihood of the given samples
        self.mu_ = X.mean()

        if X.size < 2:
            self.var_ = 0
            return self

        norm_factor = (1 / X.size) if self.biased_ else (1 / (X.size - 1))
        self.var_ = norm_factor * ((X - self.mu_) ** 2).sum()

        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        mu = self.mu_
        sigma2 = self.var_
        return (1 / np.sqrt(2 * np.pi * sigma2)) \
            * np.exp(-(X-mu)**2/(2*sigma2))

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        m = X.size
        temp_x = (X - mu) ** 2
        return (-m/2) * np.log(2 * np.pi * sigma) - (1/(2*sigma)) * temp_x.sum()


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """

        num_of_samples, sample_dim = X.shape
        self.mu_ = X.mean(axis=0)

        if num_of_samples <= 1:
            self.cov_ = np.zeros(sample_dim, sample_dim)
        else:
            x_centered = X - self.mu_
            self.cov_ = x_centered.T.dot(x_centered) / (num_of_samples - 1)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        cov_det = np.linalg.det(self.cov_)
        inverse_cov = np.linalg.inv(self.cov_)
        num_of_samples, sample_dim = X.shape

        x_centered = X - self.mu_
        factor = 1 / np.sqrt((2 * np.pi) ** sample_dim * cov_det)

        # calculate (X-mu)^T * COV^-1 * (X-mu) for each sample X
        exp_param = (x_centered.dot(inverse_cov) * x_centered).sum(-1)

        return factor * np.exp(-exp_param / 2)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        # m = number of samples, d = sample dimension
        m, d = X.shape
        inverse_cov = np.linalg.inv(cov)

        # calc log(L) = -1/2 * [m*d*log(2pi) + m*log(det(cov)) + sum_1-m{Xi-mu * cov.I * Xi-mu}]
        part1 = m * d * np.log(2 * np.pi)
        sign, log_det_cov = slogdet(cov)
        part2 = m * sign * log_det_cov
        x_centered = X - mu
        # par3 equivalent to trace(X.T dot cov.I dot X) which is also sum_1-m{Xi-mu * cov.I * Xi-mu}]
        part3 = (x_centered.dot(inverse_cov) * x_centered).sum()
        return (-1/2) * (part1 + part2 + part3)
