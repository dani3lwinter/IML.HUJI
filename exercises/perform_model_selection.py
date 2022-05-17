from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# pio.templates.default = "simple_white"
pio.renderers.default = "browser"


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    noise_std = np.sqrt(noise)
    response = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

    x = np.linspace(-1.2, 2, n_samples)
    y_noiseless = response(x)
    y = y_noiseless + np.random.normal(0, noise_std, size=len(y_noiseless))
    combined_y = np.vstack([y, y_noiseless]).T
    noisy, noiseless = 0, 1
    train_x, train_y, test_x, test_y = split_train_test(x, combined_y, 2/3)
    train_x, test_x = np.array(train_x), np.array(test_x)

    fig = go.Figure(data=[go.Scatter(x=train_x, y=train_y[:, noisy], name="True y (noiseless)", mode="markers"),
                          go.Scatter(x=test_x, y=test_y[:, noisy], name="y with noise", mode="markers")],
                    layout=go.Layout(title="True (noiseless) model of train and test samples",
                                     xaxis_title='x',
                                     yaxis_title='y = f(x)'))
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    degrees = np.arange(11)
    train_errors = np.empty(len(degrees))
    valid_errors = np.empty(len(degrees))
    for i, k in enumerate(degrees):
        estimator = PolynomialFitting(k)
        train_errors[i], valid_errors[i] =\
            cross_validate(estimator, train_x, train_y[:, noisy], mean_square_error)

    # Plot for each value of k the average training- and validation errors
    fig = go.Figure(data=[go.Scatter(x=degrees, y=train_errors, name="Training Error"),
                          go.Scatter(x=degrees, y=valid_errors, name="Validation Error")],
                    layout=go.Layout(title='Cross validation Error as function of Polynomial Degree<br>' +
                                           f'Noise Variance = {noise}, Sample Size = {n_samples}',
                                     xaxis_title='Fitted Polynomial Degree',
                                     yaxis_title='Error'))
    fig.update_xaxes(type='category')
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = degrees[np.argmin(valid_errors)]
    print(f"=== Noise Variance: {noise} ===")
    print(f"Best degree is {best_k}")
    estimator = PolynomialFitting(best_k)
    estimator.fit(train_x, train_y[:, noisy])

    # evaluate the best degree with test set
    y_pred = estimator.predict(test_x)
    y_test = test_y[:, noisy]
    loss = mean_square_error(y_test, y_pred)
    print(f"The test error with that degree is {loss: .2f}")


def run_cross_validation(estimator_ctr, X, y, theta_range, param_name):
    train_errors = np.empty(len(theta_range))
    valid_errors = np.empty(len(theta_range))
    for i, t in enumerate(theta_range):
        estimator = estimator_ctr(**{param_name: t})
        train_errors[i], valid_errors[i] = \
            cross_validate(estimator, X, y, mean_square_error)
    return train_errors, valid_errors

def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_x, train_y, test_x, test_y = split_train_test(X, y, n_samples/len(y))
    train_x, train_y = np.array(train_x), np.array(train_y)
    test_x, test_y = np.array(test_x), np.array(test_y)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.arrange(n_evaluations)
    train_errors = np.empty(len(lambdas))
    valid_errors = np.empty(len(lambdas))
    for i, k in enumerate(lambdas):
        estimator = PolynomialFitting(k)
        train_errors[i], valid_errors[i] = \
            cross_validate(estimator, train_x, train_y, mean_square_error)

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # select_polynomial_degree()
    # select_polynomial_degree(noise=0)
    # select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()