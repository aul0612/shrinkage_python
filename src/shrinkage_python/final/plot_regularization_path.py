import plotly.graph_objects as go


def plot_regularization_path(mean_coefficients, lambda_values, beta):
    """Plot the regularization path for the coefficients.

    Parameters
    ----------
    coefficients_all : np.ndarray
        3D numpy array with shape (n_lambda, n_simulations, n_features).
    lambda_values : np.ndarray
        1D numpy array with the regularization parameters.
    beta : np.ndarray
        1D numpy array with the true coefficients.
    """
    # Create a plotly figure
    fig = go.Figure()

    # Add traces for each feature
    for i in range(mean_coefficients.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=lambda_values,
                y=mean_coefficients[:, i],
                mode="lines",
                name=f"Feature {i + 1}",
            )
        )

    # Add a trace for the true coefficients
    fig.add_trace(
        go.Scatter(
            x=lambda_values,
            y=beta,
            mode="lines",
            name="True Coefficients",
            line=dict(dash="dash"),
        )
    )

    # Update layout
    fig.update_layout(
        title="Regularization Path",
        xaxis_title="Regularization Parameter (lambda)",
        yaxis_title="Coefficient Value",
        xaxis_type="log",
        legend_title_text="Features",
    )

    return fig
