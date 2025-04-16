import numpy as np
from scipy.stats import multivariate_normal
from sklearn.linear_model import Ridge


def run_simulation(seed, n, k, b, cov_x, means_x, beta, sigma, lambda_values):
    """Run simulation for regression analysis!
    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n : int
        Number of samples.
    k : int
        Number of features.
    b : int
        Number of simulations.
    cov_x : array-like
        Covariance matrix for the independent variables.
    means_x : array-like
        Means for the independent variables.
    beta : array-like
        Coefficients for the regression model.
    sigma : float
        Standard deviation of the noise.
    lambda_values : array-like
        Regularization parameters for Ridge regression.

    Returns:
    -------
    pd.DataFrame
        DataFrame containing the results of the simulation.
    """
    np.random.Generator.seed(seed)
    lam_len = len(lambda_values)
    coefficients_all = np.zeros((lam_len, b, k))
    for b_idx in range(b):
        x = multivariate_normal.rvs(mean=means_x, cov=cov_x, size=n)
        eps = np.random.Generator.normal(0, sigma, n)
        y = x @ beta + eps
        for lam_idx, lam in enumerate(lambda_values):
            ridge = Ridge(alpha=lam, fit_intercept=False)
            ridge.fit(x, y)
            coefficients_all[lam_idx, b_idx, :] = ridge.coef_
    return coefficients_all
