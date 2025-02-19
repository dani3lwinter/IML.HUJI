from typing import Tuple
import numpy as np
import pandas as pd


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    train_size = int(train_proportion * len(y))

    # select train_size random indices
    train_indices = np.random.choice(range(len(y)), train_size, replace=False)
    train_mask = np.array([False] * len(y))
    train_mask[train_indices] = True

    train_x = X[train_mask]
    train_y = y[train_mask]
    test_x = X[~train_mask]
    test_y = y[~train_mask]

    # rand_seed = np.random.randint(1, 100)
    # train_X = X.sample(frac=train_proportion, random_state=rand_seed)
    # train_y = y.sample(frac=train_proportion, random_state=rand_seed)
    # test_X = X.drop(train_X.index).sample(frac=1, random_state=rand_seed)
    # test_y = y.drop(train_y.index).sample(frac=1, random_state=rand_seed)
    # return train_X, train_y, test_X, test_y
    return train_x, train_y, test_x, test_y


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    raise NotImplementedError()
