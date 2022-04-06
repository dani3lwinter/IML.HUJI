import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
pio.renderers.default = "browser"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    X = pd.read_csv(filename, parse_dates=["Date"])
    X = X[X.Temp > -72]
    X["DayOfYear"] = X["Date"].dt.dayofyear

    return X


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_X = X[X.Country == 'Israel']
    years = israel_X["Year"].astype(str)
    fig = px.scatter(israel_X, x="DayOfYear", y="Temp", color=years,
                     labels={
                         "Temp": "Temperature (°C)",
                         "DayOfYear": "Day of Year",
                         "color": "Year"
                     },
                     title="Temperature as Function of Day Of Year")
    fig.show()

    # calculate the standard deviation by Month
    x_std = israel_X.groupby(["Month"]).std().reset_index()
    fig = px.bar(x_std, x='Month', y='Temp',
                 labels={
                     "Temp": "Standard Deviation",
                     "Month": "Month"
                 },
                 title="Standard Deviation of Temperature in each Month"
                 )
    fig.show()

    # Question 3 - Exploring differences between countries
    x_mean = X.groupby(["Country", "Month"]).mean().reset_index()
    x_std = X.groupby(["Country", "Month"]).std().reset_index()
    x_mean["std"] = x_std["Temp"]
    fig = px.line(x_mean, x="Month", y="Temp", color='Country', error_y="std",
                  labels={
                      "Temp": "Mean Temperature (°C)",
                      "Month": "Month"
                  },
                  title="Mean Temperature by Month in Each Country")
    fig.update_xaxes(type='category')
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    train_x, train_y, test_x, test_y = split_train_test(israel_X["DayOfYear"], israel_X["Temp"], 0.25)
    train_x, train_y = np.array(train_x), np.array(train_y)
    test_x, test_y = np.array(test_x), np.array(test_y)

    # For every value k in [1,10], fit a polynomial model of degree k using the training set
    degrees = np.arange(1, 11)
    losses = np.empty(degrees.size)
    for i in range(degrees.size):
        learner = PolynomialFitting(degrees[i])
        learner.fit(train_x, train_y)
        losses[i] = learner.loss(test_x, test_y)
        print("degree: %d, loss: %.2f" % (degrees[i], losses[i]))

    fig = px.bar(x=degrees, y=losses, text=np.round(losses, 2),
                 labels={
                     "x": "Degree of Fitted Polynom",
                     "y": "Loss of the model"
                 },
                 title="Loss as Function of Degree of Fitted Polynom"
                 )
    fig.update_xaxes(type='category')
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    best_k = 5
    learner = PolynomialFitting(best_k)
    train_x = israel_X["DayOfYear"]
    train_y = israel_X["Temp"]
    learner.fit(train_x, train_y)

    countries = X.Country.unique()
    losses = []
    for country in countries:
        country_x = X.loc[X.Country == country, "DayOfYear"]
        country_y = X.loc[X.Country == country, "Temp"]
        losses.append(learner.loss(country_x, country_y))

    fig = px.bar(x=countries, y=losses, text=np.round(losses, 2),
                 labels={
                     "x": "Country",
                     "y": "Loss of the model"
                 },
                 title="Loss of as Function the Country of the data"
                 )
    fig.show()



