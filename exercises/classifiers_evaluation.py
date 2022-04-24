import os.path

import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


pio.renderers.default = "browser"
pio.templates.default = 'plotly'


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(os.path.join("../datasets", f))

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def update_loss(model, x_, y_):
            losses.append(model.loss(X, y))

        perceptron = Perceptron(include_intercept=True, callback=update_loss)
        perceptron.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(1, len(losses)+1), y=losses, mode="lines", name="Loss"))
        fig.update_layout(title_text=f"Perceptron loss as function of fitting iteration: {n}",
                          xaxis_title="Iteration", yaxis_title="Loss")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(os.path.join("../datasets", f))

        # Fit models and predict over training set
        lda = LDA()
        naive_bayes = GaussianNaiveBayes()

        lda.fit(X, y)
        naive_bayes.fit(X, y)

        lda_pred = lda.predict(X)
        naive_bayes_pred = naive_bayes.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        lda_acc = accuracy(lda_pred, y)
        naive_bayes_acc = accuracy(naive_bayes_pred, y)

        color_map = {0: "red", 1: "blue", 2: "green"}
        lda_colors = np.array([color_map[i] for i in lda_pred])
        bayes_colors = np.array([color_map[i] for i in naive_bayes_pred])

        fig = make_subplots(rows=1, cols=2,
                            specs=[[{"type": "scatter"}, {"type": "scatter"}]],
                            subplot_titles=(
                                "Gaussian Naive Bayes, accuracy: {:.2f}%".format(naive_bayes_acc * 100),
                                "LDA, accuracy: {:.2f}%".format(lda_acc * 100)
                            ))
        fig.update_layout(title_text=f"Predictions on {f.split('.')[0]} dataset",
                          xaxis_title="x", yaxis_title="y")

        # Add traces for data-points setting symbols and colors
        for c in lda.classes_:
            filter = naive_bayes_pred == c
            fig.add_trace(go.Scatter(x=X[filter][:, 0], y=X[filter][:, 1],
                                     mode="markers",
                                     name=f"Class {c}",
                                     marker_color=bayes_colors[filter],
                                     marker_symbol=y[filter],
                                     legendgroup=f"group {c}"),
                          row=1, col=1)

            filter = lda_pred == c
            fig.add_trace(go.Scatter(x=X[filter][:, 0], y=X[filter][:, 1],
                                     mode="markers",
                                     name=f"class {c}",
                                     marker_color=lda_colors[filter],
                                     marker_symbol=y[filter],
                                     legendgroup=f"group {c}",
                                     showlegend=False,),
                          row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(x=naive_bayes.mu_[:, 0], y=naive_bayes.mu_[:, 1],
                                 mode="markers", name="Fitted Center",
                                 marker=dict(color="black", symbol="x"),
                                 legendgroup="center-x"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1],
                                 mode="markers", name="Fitted Center",
                                 marker=dict(color="black", symbol="x"),
                                 showlegend=False,
                                 legendgroup="center-x"
                                 ),
                      row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(lda.classes_)):
            fig.add_trace(get_ellipse(naive_bayes.mu_[i], np.diag(naive_bayes.vars_[i])),
                          row=1, col=1)
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_),
                          row=1, col=2)

        # fig.update_yaxes(
        #     scaleanchor="x",
        #     scaleratio=1,
        # )
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
