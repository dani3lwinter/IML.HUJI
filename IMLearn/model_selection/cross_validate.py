from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    unfitted_est = deepcopy(estimator)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    # split the indices to cv folds
    indices_folds = np.array_split(indices, cv)

    # get rid of empty folds
    indices_folds = [x for x in indices_folds if x.size > 0]

    train_scores = np.empty(cv)
    test_scores = np.empty(cv)

    for i in range(cv):
        # build S\S_i
        test_fold = indices_folds[i]
        train_folds = [fold for j, fold in enumerate(indices_folds) if i != j]
        train_folds = np.concatenate(train_folds)

        estimator = deepcopy(unfitted_est)
        estimator.fit(X[train_folds], y[train_folds])
        y_test_pred = estimator.predict(X[test_fold])
        y_train_pred = estimator.predict(X[train_folds])

        # evaluate the model
        test_scores[i] = scoring(y[test_fold], y_test_pred)
        train_scores[i] = scoring(y[train_folds], y_train_pred)

    return train_scores.mean(), test_scores.mean()





