import sys

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

import datetime as dt
import os

pio.templates.default = "plotly_white"
pio.renderers.default = "browser"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    # load data
    X = pd.read_csv(filename)

    # keep only rows with positive price
    X = X[X.price > 0]

    area_columns = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_lot15', 'sqft_living15']
    for col in area_columns:
        X = X[X[col] >= 0]

    # remove extreme cases
    X = X[X.price < 5_000_000]

    # convert "date" column from strings to ints
    X.date = pd.to_datetime(X.date)
    X.date = X.date.map(dt.datetime.toordinal)

    # replace empty yr_renovated cells with the value in yr_built
    X.loc[X.yr_renovated == 0, "yr_renovated"] = X.loc[X.yr_renovated == 0, "yr_built"]

    # fill empty date cells
    X.loc[X.date < 700000, "date"] = np.round(X.date.mean())

    # convert "zipcode" column to dummy
    X = pd.get_dummies(X, columns=["zipcode"])

    responses = X["price"]
    X.drop('price', axis=1, inplace=True)
    X.drop('id', axis=1, inplace=True)
    X.drop('lat', axis=1, inplace=True)
    X.drop('long', axis=1, inplace=True)

    return X, responses


def calc_correlation(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate the Pearson Correlation between x and y.
    """
    cov_xy = np.cov(x, y)[0][1]
    if np.isnan(cov_xy):
        return 0
    std_x = np.std(x)
    std_y = np.std(y)
    return cov_xy / (std_x * std_y)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for col in X.columns:
        corr = calc_correlation(X[col], y)
        fig = go.Figure(go.Scatter(x=X[col], y=y, mode='markers', marker={'size': 3}))
        fig.update_layout(title="House Prices as Function of %s. Correlation is %.3f" % (col, corr))
        fig.update_xaxes(title_text=col + " Value")
        fig.update_yaxes(title_text="House Price")
        fig.write_image(os.path.join(output_path, col + ".png"))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, r".")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    learner = LinearRegression(include_intercept=True)
    percentage = np.arange(0.1, 1.01, 0.01)
    losses_mean = np.empty(percentage.size)
    losses_std = np.empty(percentage.size)

    for i in range(len(percentage)):
        p = percentage[i]
        losses = np.empty(10)

        # fit the model 10 times in p% of the training set
        for j in range(10):
            new_X, new_y, _, _ = split_train_test(train_X, train_y, p)
            learner.fit(np.array(new_X), np.array(new_y))
            losses[j] = learner.loss(np.array(test_X), np.array(test_y))

        losses_mean[i] = losses.mean()
        losses_std[i] = losses.std()

    # plot the results
    go.Figure(
        [
            go.Scatter(x=percentage, y=losses_mean - 2 * losses_std, fill=None,
                       mode="lines", line=dict(color="lightgrey"), showlegend=False),
            go.Scatter(x=percentage, y=losses_mean + 2 * losses_std, fill='tonexty',
                       mode="lines", line=dict(color="lightgrey"), showlegend=False),
            go.Scatter(x=percentage, y=losses_mean, mode="markers+lines",
                       marker=dict(color="black", size=1), showlegend=False)
        ],
        layout=go.Layout(
            title="Mean loss as a function of p%, with confidence interval of mean(loss)±2*std(loss)",
            xaxis_title="p% - The percentage of used samples from the training set",
            yaxis_title="The mean loss over 10 times of training",
        )
    ).show()

