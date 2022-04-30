from __future__ import annotations
from typing import NoReturn, Tuple
# from ...base import BaseEstimator
from IMLearn.base import BaseEstimator
import numpy as np
from itertools import product

from IMLearn.metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        if len(X.shape) == 1:
            X = X.reshape((-1, 1))
        n_features = X.shape[1]

        best_feature = 0
        best_thr, best_sign, least_err = 0, 1, 1

        for feature, sign in product(range(n_features), [-1, 1]):
            thr, thr_err = self._find_threshold(X[:, feature], y, sign)
            if thr_err <= least_err:
                best_feature = feature
                best_thr, best_sign, least_err = thr, sign, thr_err

        self.sign_ = best_sign
        self.j_ = best_feature
        self.threshold_ = best_thr

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        if len(X.shape) == 1:
            X = X.reshape((-1, 1))

        n_samples, n_features = X.shape
        prediction = np.full(n_samples, self.sign_)
        prediction[X[:, self.j_] < self.threshold_] = -self.sign_
        return prediction

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        indx = np.argsort(values)
        values = values[indx]
        labels = labels[indx]

        best_thr = values[0]
        misses = np.sum(labels != sign)
        least_misses = misses

        # allow the threshold to be a little above all samples
        np.append(values, values[-1] + np.abs(0.01 * values[-1]))

        for i in range(1, len(values)):
            # when the threshold is value[i],
            # label[i-1] is a miss iff  label[i-1]*(-sign) = -1 iff label[i-1]*sign = 1
            misses += labels[i-1]*sign
            if misses < least_misses:
                least_misses = misses
                best_thr = values[i]

        return best_thr, least_misses/labels.size

        best_thr = 0
        least_err = 1

        for thr in np.unique(values):

            # evaluate the misclassification error for this threshold
            # prediction = np.full(labels.shape, sign)
            prediction = np.where(values < thr, -sign, sign)
            err = misclassification_error(labels, prediction, normalize=True)

            if err < least_err:
                least_err = err
                best_thr = thr

        return best_thr, least_err

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
