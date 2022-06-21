import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

import IMLearn.desent_methods.modules
from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression

from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

from utils import decision_surface
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


fixed_lr_fig_title = r"$\text{{Convergence Rate - The }}{0} \text{{ norm as a function of the GD iteration}}\\" \
                     r"\text{{Using fixed learning rate }}\eta$"

MODELS = [L1, L2]
MODEL_NAMES = [r'\ell_1', r'\ell_2^2']


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    print(f"=== Testing fixed learning rates ===")
    # etas = [0.01]
    for model, model_name in zip(MODELS, MODEL_NAMES):

        convergence_rate_fig = go.Figure(layout=go.Layout(
            xaxis_title='Number of Iteration',
            yaxis_title='Norm',
            title=fixed_lr_fig_title.format(model_name)))

        print(f"Model Name\teta\tMin Norm")
        for eta in etas:
            callback, values, weights = get_gd_state_recorder_callback()
            gd = GradientDescent(FixedLR(eta), callback=callback)
            gd.fit(model(np.array(init)), np.empty(0), np.empty(0))

            # plot descent path
            title = rf"\text{{of }}{model_name}\text{{ model, using fixed }}\eta={eta}"
            fig = plot_descent_path(model, np.array(weights), title=title)
            fig.show()

            # plot convergence rate
            convergence_rate_fig.add_trace(go.Scatter(y=values, x=np.arange(1, len(values)+1),
                                                      mode="lines", name=rf"$\eta={eta}$"))
            print(f"{model_name:<10}\t\t{eta:<6}\t{np.min(values):.3f}")

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

    print(f"=== Testing exponential decay rate ===")
    print(f" gamma  | l1 norm")
    for gamma in gammas:
        callback, values, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(ExponentialLR(eta, gamma), callback=callback)
        gd.fit(L1(np.array(init)), np.empty(0), np.empty(0))

        # plot convergence rate
        convergence_rate_fig.add_trace(go.Scatter(y=values, x=np.arange(1, len(values) + 1),
                                                  mode="lines", name=rf"$\gamma={gamma}$"))

        print(f" {gamma:<6} | \t{np.min(values)}")

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
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['chd', 'row.names'], axis=1), df.chd, train_size=train_portion)
    return X_train, y_train, X_test, y_test


def calc_roc_curve(y_true: np.ndarray, y_proba: np.ndarray):
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

    thresholds, indices = np.unique(y_proba, return_index=True)
    return fpr_vec[indices], tpr_vec[indices], thresholds


def plot_roc_curve(y_true, y_proba, title):
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
        layout=go.Layout(title=rf"$\text{{ROC Curve: {title}, AUC}}={auc(fpr, tpr):.4f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.show()

    best_alpha_index = np.argmax(tpr - fpr)
    return thresholds[best_alpha_index]


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
    y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    gd = GradientDescent(FixedLR(1e-4), max_iter=20_000)
    estimator = LogisticRegression(solver=gd)
    estimator.fit(X_train, y_train)
    best_alpha = plot_roc_curve(y_train, estimator.predict_proba(X_train),
                   title='Logistic Regression without regularization')

    print(" === Question 9 ===")
    print("Best threshold: alpha=", best_alpha)

    # fit again with the best alpha and report the test error
    estimator = LogisticRegression(solver=gd, alpha=best_alpha)
    estimator.fit(X_train, y_train)
    print(f"The misclassification rate on the test set is {estimator.loss(X_test, y_test):.4f}\n")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    gd = GradientDescent(FixedLR(1e-4), max_iter=20_000)
    for penalty in ['l1', 'l2']:
        print(f"==== Testing {penalty} penalty ====")
        lambdas = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
        train_err, valid_err = np.zeros(len(lambdas)), np.zeros(len(lambdas))

        for i, lam in enumerate(lambdas):
            estimator = LogisticRegression(solver=gd, penalty=penalty, lam=lam)
            train_err[i], valid_err[i] = cross_validate(estimator, X_train, y_train, misclassification_error)

        # plot the cv error
        fig = go.Figure(data=[go.Scatter(x=lambdas, y=train_err, name="Training Error"),
                              go.Scatter(x=lambdas, y=valid_err, name="Validation Error")],
                        layout=go.Layout(title='Cross validation Error as function of lambda, using '+penalty,
                                         xaxis_title='lambda',
                                         yaxis_title='misclassification error'))
        fig.show()

        best_lam_index = valid_err.argmin()
        print(f"Best lambda is {lambdas[best_lam_index]}")
        print(f"\tValidation Error: {valid_err[best_lam_index]}")

        # fit with selected lambda and report the test error
        best_est = LogisticRegression(solver=gd, penalty=penalty, lam=lambdas[best_lam_index])
        best_est.fit(X_train, y_train)
        print(f"\tTest Error: {best_est.loss(X_test, y_test)}\n")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
