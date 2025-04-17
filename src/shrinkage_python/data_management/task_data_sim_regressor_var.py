import numpy as np

from shrinkage_python.config import BLD, SRC
from shrinkage_python.data.params_regressor_var import (
    b,
    beta,
    cov_x,
    k,
    lambda_values,
    means_x,
    n,
    seed,
    sigma,
)
from shrinkage_python.data_management.data_sim_regressor_var import run_simulation


def task_run_simulation(
    script=SRC / "data_management" / "data_sim_regressor_var.py",
    data=SRC / "data" / "params_regressor_var.py",
    produces=BLD / "data" / "simulation_regressor_var.npy",
):
    """Run the simulation for regression analysis."""
    # Run the simulation
    coefficients_all = run_simulation(
        seed, n, k, b, cov_x, means_x, beta, sigma, lambda_values
    )

    # Save the results
    np.save(produces, coefficients_all)
