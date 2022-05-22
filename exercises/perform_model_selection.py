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


def run_cross_validation(estimator_ctr, X, y, theta_range, param_name):
    train_errors = np.empty(len(theta_range))
    valid_errors = np.empty(len(theta_range))
    for i, t in enumerate(theta_range):
        estimator = estimator_ctr(**{param_name: t})
        train_errors[i], valid_errors[i] = \
            cross_validate(estimator, X, y, mean_square_error)
    return train_errors, valid_errors


def plot_cross_validation_error(x, train_errors, valid_errors, title, xaxis_title):
    fig = go.Figure(data=[go.Scatter(x=x, y=train_errors, name="Training Error"),
                          go.Scatter(x=x, y=valid_errors, name="Validation Error")],
                    layout=go.Layout(title=title,
                                     xaxis_title=xaxis_title,
                                     yaxis_title='Error'))
    if len(x) < 20:
        fig.update_xaxes(type='category')
    fig.show()


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
    response = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

    x = np.linspace(-1.2, 2, n_samples)
    np.random.shuffle(x)
    np.random.shuffle(x)
    # np.random.shuffle(x)
    y_noiseless = response(x)
    y = y_noiseless + np.random.normal(0, noise, size=len(y_noiseless))

    train_x, train_y, test_x, test_y = split_train_test(x, y, 2/3)
    train_x, test_x = np.array(train_x), np.array(test_x)

    # train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(x), pd.Series(y), 2 / 3)
    # train_x, test_x = train_x.squeeze().to_numpy(), test_x.squeeze().to_numpy()
    # train_y, test_y = train_y.to_numpy(), test_y.to_numpy()

    x_noiseless = np.linspace(x.min(), x.max(), 200)
    fig = go.Figure(data=[go.Scatter(x=x_noiseless, y=response(x_noiseless),
                                     name="True model", mode="lines", line_color='black'),
                          go.Scatter(x=train_x, y=train_y, name="Train set", mode="markers"),
                          go.Scatter(x=test_x, y=test_y, name="Test set", mode="markers")],
                    layout=go.Layout(title="True (noiseless) model, and train and test sets",
                                     xaxis_title='x',
                                     yaxis_title='y = f(x)'))
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    degrees = np.arange(11)
    train_errors, valid_errors = run_cross_validation(PolynomialFitting,
                                                      train_x, train_y,
                                                      theta_range=degrees,
                                                      param_name='k')

    # Plot for each value of k the average training- and validation errors
    plot_cross_validation_error(degrees, train_errors, valid_errors,
                                title='Cross validation Error as function of Polynomial Degree<br>' +
                                      f'Noise Variance = {noise}, Sample Size = {n_samples}',
                                xaxis_title='Degree of Fitted Polynomial')

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = degrees[np.argmin(valid_errors)]
    print(f"=== Noise Variance: {noise} ===")
    print(f"Best degree is {best_k}")
    estimator = PolynomialFitting(best_k)
    estimator.fit(train_x, train_y)

    # evaluate the best degree with test set
    y_pred = estimator.predict(test_x)
    loss = mean_square_error(test_y, y_pred)
    print(f"The test error with that degree is {loss: .2f}")


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
    lambdas = np.linspace(0, 0.2, n_evaluations)
    # lambdas = np.logspace(-4, -0.5, n_evaluations)
    # === Ridge model ===
    train_errors, valid_errors = run_cross_validation(RidgeRegression,
                                                      train_x, train_y,
                                                      theta_range=lambdas,
                                                      param_name='lam')
    ridge_best_l = lambdas[np.argmin(valid_errors)]
    plot_cross_validation_error(lambdas, train_errors, valid_errors,
                                title='Ridge Regularization - Cross validation Error as function of lambda<br>' +
                                      f'Number of parameters = {n_evaluations}, Sample Size = {n_samples}',
                                xaxis_title='Lambda (regularization parameter)')

    # === Lasso model ===
    train_errors, valid_errors = run_cross_validation(Lasso,
                                                      train_x, train_y,
                                                      theta_range=lambdas,
                                                      param_name='alpha')

    lasso_best_l = lambdas[np.argmin(valid_errors)]
    plot_cross_validation_error(lambdas, train_errors, valid_errors,
                                title='Lasso Regularization - Cross validation Error as function of lambda<br>' +
                                      f'Number of parameters = {n_evaluations}, Sample Size = {n_samples}',
                                xaxis_title='Lambda (regularization parameter)')

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    print(" ==== Ridge model ====")
    print(f"Best lambda is {ridge_best_l: .3f}")
    estimator = RidgeRegression(lam=ridge_best_l).fit(train_x, train_y)

    # evaluate the best degree with test set
    y_pred = estimator.predict(test_x)
    loss = mean_square_error(test_y, y_pred)
    print(f"The test error with that lambda is {loss: .3f}\n")

    print(" ==== Lasso model ====")
    print(f"Best lambda is {lasso_best_l: .3f}")
    estimator = Lasso(alpha=lasso_best_l).fit(train_x, train_y)

    # evaluate the best degree with test set
    y_pred = estimator.predict(test_x)
    loss = mean_square_error(test_y, y_pred)
    print(f"The test error with that lambda is {loss: .3f}\n")

    print(" ==== Linear model ====")
    estimator = LinearRegression().fit(train_x, train_y)

    # evaluate the best degree with test set
    y_pred = estimator.predict(test_x)
    loss = mean_square_error(test_y, y_pred)
    print(f"The test error  is {loss: .3f}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
