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

    def joint_angles(self, q: np.ndarray, dt: float = 1.0, title: str = "Joint Angles"):
        """
        Plot joint angles over time as stacked subplots.
        
        Args:
            q: Array of shape (n_joints, N) containing joint angles
            dt: Time step for x-axis scaling
            title: Plot title
        """
        n_joints, N = q.shape
        t = np.arange(N) * dt

        fig = make_subplots(
            rows=n_joints, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=[f"$q_{{{i+1}}}$" for i in range(n_joints)]
        )

        for i in range(n_joints):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=q[i, :],
                    mode="lines",
                    line=dict(color=self.colors[i % len(self.colors)], width=1.5),
                    name=f"$q_{{{i+1}}}$"
                ),
                row=i + 1, col=1
            )
            fig.update_yaxes(title_text=f"$q_{{{i+1}}}$ [rad]", row=i + 1, col=1)

        fig.update_xaxes(title_text="$t$ [s]", row=n_joints, col=1)
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            template=self.template,
            height=200 * n_joints,
            showlegend=False
        )

        return fig

    def generic_plot(
        self,
        *series: np.ndarray,
        dt: float = 1.0,
        xlabel: str = "$t$",
        ylabel: str = "$y$",
        title: str = "Plot",
        labels: list = None,
    ):
        """
        Generic line plot for one or more 1D time series.
        
        Args:
            *series: One or more 1D arrays (1, N) or (N,)
            dt: Time step for x-axis scaling
            xlabel: X-axis label (supports LaTeX with $...$)
            ylabel: Y-axis label (supports LaTeX with $...$)
            title: Plot title
            labels: List of labels for each series (supports LaTeX with $...$)
        """
        fig = go.Figure()

        for i, s in enumerate(series):
            s = np.atleast_1d(np.squeeze(s))
            t = np.arange(len(s)) * dt
            
            label = labels[i] if labels and i < len(labels) else f"$S_{{{i+1}}}$"
            color = self.colors[i % len(self.colors)]

            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=s,
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

    def save(self, fig: go.Figure, filename: str, scale: int = 2):
        """
        Save figure to file. Supports png, jpg, svg, pdf, html.
        
        Args:
            fig: Plotly figure
            filename: Output filename with extension
            scale: Scale factor for image resolution (ignored for html)
        """
        if filename.endswith(".html"):
            fig.write_html(filename, include_mathjax="cdn")
        else:
            fig.write_image(filename, scale=scale)
