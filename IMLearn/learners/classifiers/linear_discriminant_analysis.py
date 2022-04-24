from typing import NoReturn

from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.mu_ = np.zeros((n_classes, n_features))
        self.cov_ = np.zeros((n_features, n_features))
        self.pi_ = np.zeros(n_classes)
        X_centered = X.copy()

        for i, cls in enumerate(self.classes_):
            class_filter = y == cls
            self.pi_[i] = class_filter.sum() / n_samples
            self.mu_[i] = np.mean(X[class_filter], axis=0)
            X_centered[class_filter] = X[class_filter] - self.mu_[i]

        self.cov_ = X_centered.T @ X_centered / (n_samples - n_classes)
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def __gaussian_pdf(self, x: np.ndarray, class_index: int) -> np.ndarray:
        """
        Calculates the pdf of a multivariate gaussian distribution of each sample in x

        Parameters
        ----------
        x : ndarray of shape (n_samples,n_features)
            Input data to calculate the probability density function for

        class_index : int the index of the class to calculate the pdf for

        Returns
        -------
        pdf : ndarray of shape (n_samples,)
            The probability density function of the multivariate gaussian distribution
        """
        mu, cov, cov_inv = self.mu_[class_index], self.cov_, self._cov_inv

        n_samples, n_features = x.shape
        x_centered = x - mu
        factor = np.sqrt(np.power(2 * np.pi, n_features) * det(cov))
        pdf = np.array([np.exp(-0.5 * sample.T @ cov_inv @ sample) / factor
                        for sample in x_centered])

        return pdf

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        n_classes = len(self.classes_)
        likelihoods = np.vstack([self.__gaussian_pdf(X, i) * self.pi_[i]
                                 for i in range(n_classes)]).T
        return likelihoods

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))



