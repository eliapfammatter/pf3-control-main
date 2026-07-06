"""
Plotting utilities for FMI co-simulation results.
"""

from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_fmi_results(
    timestamps,
    inputs,
    outputs,
    title="FMI Co-simulation Results",
    output_file=None,
    show=True,
):
    """
    Create interactive Plotly plot for FMI co-simulation results.

    Parameters
    ----------
    timestamps : list or array
        Time values
    inputs : dict
        Input variables: {name: [values]}
    outputs : dict
        Output variables: {name: [values]}
    title : str, optional
        Plot title. Default: "FMI Co-simulation Results"
    output_file : str or Path, optional
        Path to save HTML file. If None, generates timestamped filename.
    show : bool, optional
        Whether to show plot in browser. Default: True

    Returns
    -------
    str
        Path to saved HTML file
    """
    num_inputs = len(inputs)
    num_outputs = len(outputs)
    num_total = num_inputs + num_outputs

    # Combine subplot titles
    subplot_titles = [f"Input: {name}" for name in inputs.keys()] + [
        f"Output: {name}" for name in outputs.keys()
    ]

    fig = make_subplots(
        rows=num_total,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
    )

    # Color palette
    colors = ["blue", "red", "green", "orange", "purple", "brown", "cyan", "magenta"]
    row_num = 1

    # Add trace for each input variable
    for i, (name, values) in enumerate(inputs.items()):
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                mode="lines",
                name=name,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f"Time: %{{x:.2f}}s<br>{name}: %{{y:.3f}}<extra></extra>",
            ),
            row=row_num,
            col=1,
        )
        row_num += 1

    # Add trace for each output variable
    for i, (name, values) in enumerate(outputs.items()):
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                mode="lines",
                name=name,
                line=dict(color=colors[(i + num_inputs) % len(colors)], width=2),
                hovertemplate=f"Time: %{{x:.2f}}s<br>{name}: %{{y:.3f}}<extra></extra>",
            ),
            row=row_num,
            col=1,
        )
        row_num += 1

    # Update axes labels
    fig.update_xaxes(title_text="Time [s]", row=num_total, col=1)
    row_num = 1
    for name in inputs.keys():
        fig.update_yaxes(title_text=name, row=row_num, col=1)
        row_num += 1
    for name in outputs.keys():
        fig.update_yaxes(title_text=name, row=row_num, col=1)
        row_num += 1

    fig.update_layout(
        title_text=title,
        height=250 * num_total,
        showlegend=True,
        hovermode="x unified",
    )

    # Generate output filename
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"fmi_results_{timestamp}.html"

    # Save HTML
    fig.write_html(output_file)
    print(f"Interactive plot saved to: {output_file}")

    # Show in browser
    if show:
        fig.show()

    return output_file
