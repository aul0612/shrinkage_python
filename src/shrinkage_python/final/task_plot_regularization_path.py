import pandas as pd

from shrinkage_python.config import BLD, SRC
from shrinkage_python.data.params_regressor_var import beta, lambda_values
from shrinkage_python.final.plot_regularization_path import plot_regularization_path


def task_plot_regularization_path(
    script=SRC / "final" / "plot_regularization_path.py",
    data=BLD / "data" / "summary_statistics_regressor_var.pickle",
    produces=BLD / "figures" / "regularization_path.svg",
):
    """"""
    mean_coefficients = pd.read_pickle(data).pivot(
        index="Lambda", columns="Feature", values="Mean Coefficients"
    )
    fig = plot_regularization_path(mean_coefficients.to_numpy(), lambda_values, beta)
    fig.write_image(BLD / "figures" / "regularization_path.svg")
