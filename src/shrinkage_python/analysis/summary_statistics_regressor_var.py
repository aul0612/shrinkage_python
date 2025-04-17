import numpy as np
import pandas as pd


def summary_statistics(data, beta, lambda_values, k):
    """Takes a 3D numpy array and returns a 2D numpy array with summary statistics.
    Parameters
    ----------
    data : np.ndarray
        3D numpy array with shape (n_lambda, n_simulations, n_features).
    """
    bias_squared = np.zeros((len(lambda_values), k))
    variance = np.zeros((len(lambda_values), k))
    mse = np.zeros((len(lambda_values), k))
    mean_coefficients = np.zeros((len(lambda_values), k))
    for idx, lamb in enumerate(lambda_values):
        bias_squared[idx, :] = (np.mean(data[idx, :, :], axis=0) - beta) ** 2
        variance[idx, :] = np.var(data[idx, :, :], axis=0)
        mse[idx, :] = bias_squared[idx, :] + variance[idx, :]
        mean_coefficients[idx, :] = np.mean(data[idx, :, :], axis=0)

    # Reshape the results to 2D arrays
    result = pd.DataFrame(
        {
            "Lambda": np.repeat(lambda_values, k),
            "Feature": np.tile(np.arange(1, k + 1), len(lambda_values)),
            "Bias^2": bias_squared.flatten(),
            "Variance": variance.flatten(),
            "MSE": mse.flatten(),
            "Mean Coefficients": mean_coefficients.flatten(),
        }
    )
    return result
