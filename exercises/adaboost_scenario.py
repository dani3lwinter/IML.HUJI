from itertools import product

import numpy as np
from typing import Tuple
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metalearners import AdaBoost
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default = "browser"


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def add_partial_decision_boundary(fig, X, y, t, learner, lims, row=None, col=None):
    """
    Plot the decision boundary of ensemble with t estimators
    """
    # symbols = np.array(["circle", "x"])[((y + 1) / 2).astype(int)]
    predict = lambda X_: learner.partial_predict(X_, t)
    accuracy = 1 - learner.partial_loss(X, y, t)

    fig.add_trace(decision_surface(predict, lims[0], lims[1], showscale=False),
                  row=row, col=col)

    class0 = y == -1
    fig.add_trace(go.Scatter(x=X[class0][:, 0], y=X[class0][:, 1], mode="markers",
                             name="Class -1", legendgroup='Class -1', showlegend=False,
                             marker=dict(color="red", symbol="circle", line=dict(color="black", width=1))),
                  row=row, col=col)

    class1 = y == 1
    fig.add_trace(go.Scatter(x=X[class1][:, 0], y=X[class1][:, 1], mode="markers",
                             name="Class 1", legendgroup='Class 1', showlegend=False,
                             marker=dict(color="blue", symbol="x", line=dict(color="black", width=1))),
                  row=row, col=col)

    fig.update_xaxes(title_text="x", row=row, col=col)
    fig.update_yaxes(title_text="y", row=row, col=col)
    if row is None:
        fig.update_layout(title_text=f"Decision boundary of ensemble with {t} estimators, Accuracy: {accuracy:.3f}")
    else:
        fig.layout.annotations[2*(row-1)+col-1].update(text=f"Using {t} estimators, Accuracy: {accuracy: .2f}")

    return fig


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    learner = AdaBoost(DecisionStump, n_learners)
    learner.fit(train_X, train_y)

    # Plot the training- and test errors as a function of the number of fitted learners
    num_of_learners = np.arange(1, n_learners + 1)
    train_errors = [learner.partial_loss(train_X, train_y, t) for t in num_of_learners]
    test_errors = [learner.partial_loss(test_X, test_y, t) for t in num_of_learners]
    fig = go.Figure(data=[go.Scatter(x=num_of_learners, y=train_errors, name="Training Error"),
                          go.Scatter(x=num_of_learners, y=test_errors, name="Test Error")],
                    layout=go.Layout(title='Training and test errors as a function of the number of fitted learners',
                                     xaxis_title='Number of learners',
                                     yaxis_title='Error'))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [[5, 50], [100, 250]]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(rows=2, cols=2, specs=2 * [2 * [{"type": "scatter"}]],
                        subplot_titles=4*["Decision surface"],
                        vertical_spacing=0.15,
                        horizontal_spacing=0.10)

    fig.update_layout(title_text=f"Decision boundary of the ensemble",
                      xaxis_title="x", yaxis_title="y", legend_title_text='Test Set',
                      margin_t=50)

    for row, col in product(range(2), range(2)):
        add_partial_decision_boundary(fig, test_X, test_y, T[row][col], learner, lims, row=row+1, col=col+1)

    fig.data[1].showlegend = True
    fig.data[2].showlegend = True
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_T = num_of_learners[np.argmin(test_errors)]
    fig = go.Figure(layout=go.Layout(legend_title_text='Test Set'))
    add_partial_decision_boundary(fig, test_X, test_y, best_T, learner, lims)
    fig.data[1].showlegend = True
    fig.data[2].showlegend = True
    fig.show()

    # Question 4: Decision surface with weighted samples
    fig = go.Figure(layout=go.Layout(title=f'Decision boundary of fitted model, with train set',
                                     xaxis=dict(title='x'),
                                     yaxis=dict(title='y')))

    fig.add_trace(decision_surface(learner.predict, lims[0], lims[1], showscale=False))
    max_bubble_size = 50 if noise == 0 else 5
    sizeref = 2. * max(learner.D_) / (max_bubble_size ** 2)
    fig.add_trace(go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                             marker=dict(color=train_y,
                                         colorscale=[custom[0], custom[-1]],
                                         size=learner.D_,
                                         sizemode='area',
                                         sizeref=sizeref,
                                         sizemin=0.5
                                         )
                             ))
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
