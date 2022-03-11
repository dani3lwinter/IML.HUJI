from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
# pio.templates.default = "simple_white"
pio.templates.default = "plotly_white"

pio.renderers.default = "browser"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    expectation, variance = 10, 1
    sample_size = 1000
    samples = np.random.normal(expectation, variance, sample_size)
    uni_gaussian_fitter = UnivariateGaussian().fit(samples)
    print("(%f, %f)" % (uni_gaussian_fitter.mu_, uni_gaussian_fitter.var_))

    # Question 2 - Empirically showing sample mean is consistent
    sample_sizes = np.arange(10, sample_size+1, 10)
    error_distances = np.empty(sample_sizes.size)
    for i in range(sample_sizes.size):
        part_of_sample = sample_sizes[i]
        uni_gaussian_fitter.fit(samples[0:part_of_sample+1])
        error_distances[i] = np.abs(uni_gaussian_fitter.mu_ - expectation)

    fig = go.Figure(data=go.Scatter(x=sample_sizes, y=error_distances))
    fig.update_layout(title="The absolute distance between the estimated- and true value of the expectation")
    fig.update_xaxes(title_text="Sample Size",)
    fig.update_yaxes(title_text="Distance between the estimated and true expectation")
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    sample_pdf = uni_gaussian_fitter.pdf(samples)
    pdf_fig = go.Figure(data=go.Scatter(x=samples, y=sample_pdf, mode='markers'))
    pdf_fig.update_layout(title="The empirical PDF function of each sample")
    pdf_fig.update_xaxes(title_text="Sample Value", )
    pdf_fig.update_yaxes(title_text="Fitted PDF Value of the Sample")
    pdf_fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = np.array([0, 0, 4, 0])
    cov = np.array([
        [1,  0.2, 0, 0.5],
        [0.2, 2,  0,  0],
        [0,   0,  1,  0],
        [0.5, 0,  0,  1]
    ])
    sample_size = 1000
    samples = np.random.multivariate_normal(mean, cov, sample_size)
    fitter = MultivariateGaussian().fit(samples)
    print("Estimated expectation:")
    print(fitter.mu_)
    print("Estimated covariance matrix:")
    print(fitter.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    z_values = np.empty((f1.size, f3.size))

    for i in range(f1.size):
        for j in range(f3.size):
            mu = np.array([f1[i], 0, f3[j], 0])
            z_values[i][j] = MultivariateGaussian.log_likelihood(mu, cov, samples)

    heat_fig = go.Figure(go.Heatmap(x=f3, y=f1, z=z_values))
    heat_fig.update_layout(title="Question 5 - log-likelihood of mu=[f1, 0, f3, 0]")
    heat_fig.update_xaxes(title_text="f3")
    heat_fig.update_yaxes(title_text="f1")

    # label the maximum point on the graph
    f1_max, f3_max = np.unravel_index(z_values.argmax(), z_values.shape)
    label = "\n argmax: (f3=%0.3f, f1=%0.3f)" % (f3[f3_max], f1[f1_max])
    max_point = go.Scatter(x=[f3[f3_max]], y=[f1[f1_max]], text=[label],
                           mode='markers+text', textposition="bottom center")
    heat_fig.add_trace(max_point)
    heat_fig.show()

    # Question 6 - Maximum likelihood
    print("Question 6 - Maximum likelihood:")
    print(label)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
