from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

from IMLearn.learners.gaussian_estimators import UnivariateGaussian

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
        self.vars_ = np.zeros((n_classes, n_features))
        self.pi_ = np.zeros(n_classes)

        for i in range(self.classes_.shape[0]):
            class_filter = y == self.classes_[i]
            self.mu_[i] = np.mean(X[class_filter], axis=0)
            self.vars_[i] = np.var(X[class_filter], axis=0, ddof=0)
            self.pi_[i] = np.sum(class_filter) / y.shape[0]

        self.fitted_ = True

        # self.classes_ = np.unique(y)
        # for i in range(len(self.classes_)):
        #     filter_class = y == self.classes_[i]
        #     estimator = UnivariateGaussian(biased_var=True).fit(X[filter_class])
        #     self.mu_[i] = estimator.mu_
        #     self.vars_[i] = estimator.var_

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
        if not self.fitted_:
            raise RuntimeError("Estimator must first be fitted before calling `predict` function")
        likelihoods = self.likelihood(X)
        return self.classes_[np.argmax(likelihoods, axis=1)]
        np.argmin(np.log(self.pi_) + np.sum(np.log(np.sqrt(2 * np.pi * self.vars_)) + (X - self.mu_) ** 2 / (2 * self.vars_), axis=1))
        return self.classes_[ np.argmin(
            np.log(self.pi_) + np.sum(np.log(np.sqrt(2 * np.pi * self.vars_))
                                      + (X - self.mu_) ** 2 / (2 * self.vars_), axis=1))]


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

        likelihoods = np.zeros((X.shape[0], self.classes_.shape[0]))
        for i in range(self.classes_.shape[0]):
            likelihoods[:, i] = self.pi_[i] * (1/np.sqrt(2 * np.pi * self.vars_))\
                                * np.exp(-np.sum((X - self.mu_[i]) ** 2, axis=1) / (2 * self.vars_[i]))
            likelihoods[:, i] = - 0.5 * np.log(2 * np.pi * self.vars_[i])\
                                - (X - self.mu_[i]) ** 2 / (2 * self.vars_[i])\
                                + np.log(self.pi_[i])

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
        raise NotImplementedError()
