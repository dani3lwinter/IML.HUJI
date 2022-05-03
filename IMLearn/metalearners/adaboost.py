import numpy as np
# from ...base import BaseEstimator
from IMLearn.base import BaseEstimator
from typing import Callable, NoReturn

from IMLearn.metrics import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = [], np.zeros(iterations), np.zeros(iterations)
        self.D_ = np.zeros(iterations)

    @staticmethod
    def __resample(cur_X, cur_y, sample_weights):
        """
        Resample the current dataset according to sample_weights
        """
        new_indices = np.random.choice(cur_y.size, size=cur_y.size, p=sample_weights)
        return cur_X[new_indices, :], cur_y[new_indices]

    @staticmethod
    def weighted_loss(y_true, y_pred, sample_weights):
        """
        Calculate the weighted loss for a given set of predictions
        """
        misses = np.where(y_true != y_pred, 1, 0)
        return np.sum(misses * sample_weights)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        if len(X.shape) == 1:
            X = X.reshape((-1, 1))
        n_samples = X.shape[0]

        self.models_ = [self.wl_() for _ in range(self.iterations_)]
        self.D_ = np.full(n_samples, 1 / n_samples)

        cur_X, cur_y = X, y

        for i in range(self.iterations_):
            print(f"Fitting model {i+1}/{self.iterations_}", end="\r")
            self.models_[i].fit(cur_X, cur_y*self.D_)

            y_pred = self.models_[i].predict(X)
            epsilon = self.weighted_loss(y, y_pred, self.D_)

            # TODO: what to do if epsilon==0
            self.weights_[i] = np.log(1 / epsilon - 1) / 2  # if cur_loss > 0 else 1

            self.D_ = self.D_ * np.exp(-y * self.weights_[i] * y_pred)
            self.D_ = self.D_ / np.sum(self.D_)

            # Resample
            # new_indices = np.random.choice(n_samples, size=n_samples, p=self.D_)
            # cur_X, cur_y = X[new_indices, :], y[new_indices]

    def _predict(self, X):
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
        return self.partial_predict(X, self.iterations_)

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
        return misclassification_error(y, self._predict(X), normalize=True)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        selected_models = self.models_[:T]
        all_learners_pred = np.array([m.predict(X) for m in selected_models]).T
        weighted_pred = all_learners_pred @ self.weights_[:T]
        if len(weighted_pred.shape) == 1:
            return np.sign(weighted_pred)
        else:
            return np.sign(weighted_pred.sum(axis=1))

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.partial_predict(X, T)
        return misclassification_error(y, y_pred, normalize=True)
