import numpy as np

seed = 374
n = 50  # sample size
k = 20  # number of regressors
b = 1000  # number of simulations
sigma = 2  # std of noise

beta = np.ones(k)
means_x = np.zeros(k)
VARS_X = np.linspace(0.1, 5, k)
cov_x = np.diag(VARS_X)

lambda_values = np.concatenate(
    [np.linspace(0, 20, 100), np.linspace(20, 12000, 401)[1:]]
)
