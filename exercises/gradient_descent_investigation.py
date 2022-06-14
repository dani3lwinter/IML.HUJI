import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

import IMLearn.desent_methods.modules
from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
pio.templates.default = "plotly_white"


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=rf"$\text{{GD Descent Path }} {title}$"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    # def callback(gd, w, val, grad, t, eta, delta):
    def callback(gd, **kwargs):
        values.append(np.array(kwargs['val']))
        weights.append(np.array(kwargs['weights']))

    return callback, values, weights


fixed_lr_fig_title = r"$\text{{Convergence Rate - The }}{0} \text{{ norm as a function of the GD iteration}}<br>" \
                     r"\text{{Using fixed learning rate }}\eta$"

MODELS = [L1, L2]
MODEL_NAMES = [r'\ell_1', r'\ell_2^2']
# MODELS = [IMLearn.desent_methods.modules.LogisticModule]
# MODEL_NAMES = [r'logistic']

def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):

    # etas = [0.01]
    for model, model_name in zip(MODELS, MODEL_NAMES):

        convergence_rate_fig = go.Figure(layout=go.Layout(
            xaxis_title='Number of Iteration',
            yaxis_title='Norm',
            title=fixed_lr_fig_title.format(model_name)))

        for eta in etas:
            callback, values, weights = get_gd_state_recorder_callback()
            gd = GradientDescent(FixedLR(eta), callback=callback)
            gd.fit(model(np.array(init)), np.empty(0), np.empty(0))

            # plot descent path
            title = rf"\text{{of }}{model_name}\text{{ model, using fixed }}\eta={eta}"
            fig = plot_descent_path(model, np.array(weights), title=title)
            # fig.show()

            # plot convergence rate
            convergence_rate_fig.add_trace(go.Scatter(y=values, x=np.arange(1, len(values)+1),
                                                      mode="lines", name=rf"$\eta={eta}$"))

        convergence_rate_fig.show()


exp_lr_fig_title = r"$\text{Convergence Rate - The }\ell_1\text{ norm as a function of the GD iteration}\\  " \
                   r"\text{Using exponential decay rates - }\eta\cdot\gamma^t$"


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate

    convergence_rate_fig = go.Figure(layout=go.Layout(
        xaxis_title=r'$\text{Number of Iteration }(t)$',
        yaxis_title=r'$\ell_1\text{ Norm}$',
        title=exp_lr_fig_title))

    for gamma in gammas:
        callback, values, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(ExponentialLR(eta, gamma), callback=callback)
        gd.fit(L1(np.array(init)), np.empty(0), np.empty(0))

        # plot convergence rate
        convergence_rate_fig.add_trace(go.Scatter(y=values, x=np.arange(1, len(values) + 1),
                                                  mode="lines", name=rf"$\gamma={gamma}$"))

    # Plot algorithm's convergence for the different values of gamma
    convergence_rate_fig.show()

    # Plot descent path for gamma=0.95
    gamma = 0.95
    for model, model_name in zip(MODELS, MODEL_NAMES):
        callback, values, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(ExponentialLR(eta, gamma), callback=callback)
        gd.fit(model(np.array(init)), np.empty(0), np.empty(0))

        # plot descent path
        title = rf"\text{{of }}{model_name}\text{{ model, using exponential decay rates - }}\eta\cdot0.95^t"
        fig = plot_descent_path(model, np.array(weights), title=title)
        fig.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
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
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def my_plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, title: str):
    """
    Plot ROC curve for a given set of predictions and true labels

    Parameters:
    -----------
    y_proba : np.ndarray of shape (n_samples, )
        Predictions of the model

    y_true : np.ndarray of shape (n_samples, )
        True labels of the samples

    title : str
        Title of the plot
    """
    sorted_indices = np.argsort(y_proba)
    y_proba = y_proba[sorted_indices]
    y_true = y_true[sorted_indices]

    tpr_vec = np.ones(len(y_proba) + 1)
    fpr_vec = np.ones(len(y_proba) + 1)

    tn, fn = 0, 0
    tp = y_true.sum()
    fp = len(y_true) - tp

    for i, alpha in enumerate(y_proba):
        # when alpha = y_proba[i], y_pred[i] flips from 1 to 0
        if y_true[i] == 0:
            fp -= 1
            tn += 1
            fpr_vec[i+1] = fp / (fp + tn)
            tpr_vec[i+1] = tpr_vec[i]
        else:
            tp -= 1
            fn += 1
            tpr_vec[i+1] = tp / (tp + fn)
            fpr_vec[i+1] = fpr_vec[i]

    fig = go.Figure(data=[go.Scatter(x=fpr_vec, y=tpr_vec, mode="lines", name="ROC curve"),
                          go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(color="black", dash='dash'),
                                     name="Random Class Assignment")
                          ],
                    layout=go.Layout(title=title, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate"))
    fig.show()


def plot_roc_curve(y_true, y_proba):
    """
    Plot ROC curve for a given set of predictions and true labels
    """
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines',
                         text=thresholds, name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.show()

def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    from sklearn.metrics import roc_curve, auc

    estimator = LogisticRegression(penalty='l2')
    # from sklearn.linear_model import LogisticRegression as LR
    # estimator = LR(penalty='none', solver='newton-cg')
    estimator.fit(X_train, y_train)
    plot_roc_curve(y_test, estimator.predict_proba(X_test))
    # plot_roc_curve(y_test, estimator.predict_proba(X_test)[:, 0])

    # plot_roc_curve(np.array(y_test), estimator.predict_proba(X_test)[:,0], "My roc")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
