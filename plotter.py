import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Plotter:
    def __init__(self, template: str = "plotly_dark"):
        self.template = template
        self.colors = [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
            "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
        ]

    def joint_angles(self, t: np.ndarray, q: np.ndarray, title: str = "Joint Angles"):
        """
        Plot joint angles over time as stacked subplots.
        
        Args:
            t: Array of shape (N,) containing time
            q: Array of shape (q.shape[0], N) containing joint angles
            title: Plot title
        """

        fig = make_subplots(
            rows=q.shape[0], cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=[f"$q_{{{i+1}}}$" for i in range(q.shape[0])]
        )

        for i in range(q.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=q[i, :],
                    mode="lines+markers",
                    line=dict(color=self.colors[i % len(self.colors)], width=1.5),
                    name=f"$q_{{{i+1}}}$"
                ),
                row=i + 1, col=1
            )
            fig.update_yaxes(title_text=f"$q_{{{i+1}}}$ [rad]", row=i + 1, col=1)

        fig.update_xaxes(title_text="$t$ [s]", row=q.shape[0], col=1)
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            template=self.template,
            height=200 * q.shape[0],
            showlegend=False
        )

        return fig

    def generic_plot(
        self,
        x: np.ndarray,
        *y_series: np.ndarray,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
        title: str = "Plot",
        labels: list = None,
    ):
        """
        Generic line plot for one or more 1D series against a common x-axis.
        
        Args:
            x: 1D array for x-axis values
            *y_series: One or more 1D arrays (1, N) or (N,) for y-axis
            xlabel: X-axis label (supports LaTeX with $...$)
            ylabel: Y-axis label (supports LaTeX with $...$)
            title: Plot title
            labels: List of labels for each series (supports LaTeX with $...$)
        """
        fig = go.Figure()
        x = np.atleast_1d(np.squeeze(x))

        for i, y in enumerate(y_series):
            y = np.atleast_1d(np.squeeze(y))
            
            label = labels[i] if labels and i < len(labels) else f"$S_{{{i+1}}}$"
            color = self.colors[i % len(self.colors)]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    name=label
                )
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            template=self.template,
            height=500,
            width=1000,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    def show(self, fig: go.Figure):
        fig.show()
