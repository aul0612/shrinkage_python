import numpy as np
import pandas as pd

from shrinkage_python.analysis.summary_statistics_regressor_var import (
    summary_statistics,
)
from shrinkage_python.config import BLD, SRC
from shrinkage_python.data.params_regressor_var import beta, k, lambda_values


def task_summary_statistics(
    script=SRC / "analysis" / "summary_statistics_regressor_var.py",
    data=BLD / "data" / "simulation_regressor_var.npy",
    produces=BLD / "data" / "summary_statistics_regressor_var.pickle",
):
    """Calculate summary statistics for the simulation results."""
    # Load the simulation results
    coefficients_all = np.load(data)

    # Calculate summary statistics
    summary_stats = summary_statistics(coefficients_all, beta, lambda_values, k)
    """summary_stats = summary_stats.melt(id_vars = ["Feature", "Lambda"],
                                       value_vars = ["Bias^2", "Variance", "MSE", "Mean Coefficients"],
                                       var_name = "Statistic",
                                       value_name = "Value")"""

    # Save the results
    pd.DataFrame(summary_stats).to_pickle(
        BLD / "data" / "summary_statistics_regressor_var.pickle"
    )
