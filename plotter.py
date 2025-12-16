import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats # For histogram bin calculation

class Plotter:
    def __init__(self, template: str = "ggplot2"):
        self.template = template

    def joints(self, t: np.ndarray, q: np.ndarray, title: str = "Joint Angles", name : str = "q", unit: str = "rad"):
        """
        Plot joint angles over time as stacked subplots.
        
        Args:
            t: Array of shape (N,) containing time
            q: Array of shape (n_joints, N) containing joint angles
            title: Plot title
        """
        fig = make_subplots(
            rows=q.shape[0], cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02
        )

        for i in range(q.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=q[i, :],
                    mode="lines",
                    line=dict(width=1.5),
                    name=f"${name}_{{{i+1}}}$"
                ),
                row=i + 1, col=1
            )
            fig.update_yaxes(title_text=f"${name}_{{{i+1}}}$ [{unit}]", row=i + 1, col=1)

        fig.update_xaxes(title_text="$t \\ [\\text{s}]$", row=q.shape[0], col=1)
        fig.update_layout(
            title=dict(text=title, font=dict(size=14)),
            template=self.template,
            autosize=True,
            height=100 * q.shape[0] + 60,
            margin=dict(l=50, r=20, t=40, b=40),
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
        xlog: bool = False,
        ylog: bool = False,
        discrete: bool = False,
    ):
        """
        Generic line plot for one or more 1D series against a common x-axis.
        
        Args:
            x: 1D array for x-axis values
            *y_series: One or more 1D arrays for y-axis
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            labels: List of labels for each series
            xlog: Use logarithmic scale for x-axis
            ylog: Use logarithmic scale for y-axis
            discrete: If True, use 'hv' line shape (for step/discrete data like iterations)
        """
        fig = go.Figure()
        x = np.atleast_1d(np.squeeze(x))
        
        line_shape = 'hv' if discrete else 'linear'

        for i, y in enumerate(y_series):
            y = np.atleast_1d(np.squeeze(y))
            label = labels[i] if labels and i < len(labels) else f"$S_{{{i+1}}}$"
            
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(width=1.5, shape=line_shape),
                    name=label,
                )
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=14)),
            xaxis=dict(
                title=dict(text=xlabel),
                type="log" if xlog else None,
            ),
            yaxis=dict(
                title=dict(text=ylabel),
                type="log" if ylog else None,
            ),
            template=self.template,
            autosize=True,
            height=280,
            margin=dict(l=50, r=120, t=40, b=40),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.8)",
            )
        )

        return fig

    def gen_html_report(
        self,
        task_figs: list = None,
        solver_figs: list = None,
        video_folder: str = None,
        title: str = "Analysis Report",
        filename: str = "report.html",
    ):
        """
        Generate HTML report with Video, Task Performance, and Solver Performance sections.
        """
        import plotly.io as pio
        import os
        import webbrowser
        
        sections_html = []
        plotly_included = False
        plot_counter = 0
        
        # Video section
        if video_folder and os.path.isdir(video_folder):
            video_files = sorted([f for f in os.listdir(video_folder) if f.lower().endswith('.mp4')])
            if video_files:
                video_cells = []
                for video_file in video_files:
                    video_path = os.path.abspath(os.path.join(video_folder, video_file))
                    video_name = os.path.splitext(video_file)[0]
                    video_html = f'''<h4>{video_name}</h4><video controls loop muted playsinline><source src="file://{video_path}" type="video/mp4"></video>'''
                    video_cells.append(f'<div class="cell">{video_html}</div>')
                
                sections_html.append(f'''
                <section>
                    <h2>Video</h2>
                    <div class="grid">
                        {"".join(video_cells)}
                    </div>
                </section>''')
        
        # Task Performance section
        if task_figs:
            task_cells = []
            for fig in task_figs:
                div_id = f"plot-{plot_counter}"
                plot_counter += 1
                include_js = 'cdn' if not plotly_included else False
                if include_js == 'cdn':
                    plotly_included = True
                html = pio.to_html(fig, full_html=False, include_plotlyjs=include_js, include_mathjax=False, div_id=div_id, config={'responsive': True})
                task_cells.append(f'<div class="cell">{html}</div>')
            
            sections_html.append(f'''
            <section>
                <h2>Task Performance</h2>
                <div class="grid">
                    {"".join(task_cells)}
            </div>
            </section>''')
        
        # Solver Performance section
        if solver_figs:
            solver_cells = []
            for fig in solver_figs:
                div_id = f"plot-{plot_counter}"
                plot_counter += 1
                include_js = 'cdn' if not plotly_included else False
                if include_js == 'cdn':
                    plotly_included = True
                html = pio.to_html(fig, full_html=False, include_plotlyjs=include_js, include_mathjax=False, div_id=div_id, config={'responsive': True})
                solver_cells.append(f'<div class="cell">{html}</div>')
            
            sections_html.append(f'''
            <section>
                <h2>Solver Performance</h2>
                <div class="grid">
                    {"".join(solver_cells)}
                </div>
            </section>''')
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; padding: 24px; background: #fafafa; font-family: system-ui, sans-serif; }}
        h1 {{ font-size: 24px; margin: 0 0 24px 0; }}
        section {{ margin-bottom: 32px; }}
        h2 {{ font-size: 18px; margin: 0 0 12px 0; }}
        h4 {{ font-size: 13px; margin: 0 0 8px 0; font-weight: 500; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 12px; }}
        .cell {{ background: #fff; border-radius: 6px; padding: 12px; min-width: 0; }}
        .cell .js-plotly-plot, .cell .plotly-graph-div {{ width: 100% !important; }}
        video {{ width: 100%; border-radius: 4px; display: block; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {"".join(sections_html)}
    <script>
        function resizePlots() {{
            document.querySelectorAll('.js-plotly-plot').forEach(p => Plotly.Plots.resize(p));
        }}
        window.addEventListener('resize', resizePlots);
        window.addEventListener('load', resizePlots);
    </script>
</body>
</html>"""
        
        filepath = os.path.abspath(filename)
        with open(filepath, "w") as f:
            f.write(html_content)
        
        webbrowser.open(f"file://{filepath}")

    def show(self, fig: go.Figure):
        """Show figure in browser."""
        fig.show(renderer="browser", config={'mathjax': 'cdn'})