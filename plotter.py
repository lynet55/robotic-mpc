import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Plotter:
    def __init__(self, template: str = "ggplot2"):
        self.template = template

    def joints(self, t: np.ndarray, *q_series: np.ndarray, labels: list = None, title: str = "Joint Angles", name : str = "q", unit: str = "rad", lower_bounds: np.ndarray = None, upper_bounds: np.ndarray = None):
        """
        Plot joint data over time for one or more simulations.
        
        Args:
            t: Array of shape (N,) containing time
            *q_series: One or more arrays of shape (n_joints, N) containing joint data.
            labels: List of labels for each series.
            title: Plot title
            name: Symbolic name for the variable (e.g., 'q').
            unit: Unit of the variable.
            lower_bounds: Optional array of shape (n_joints,) for lower boundary lines.
            upper_bounds: Optional array of shape (n_joints,) for upper boundary lines.
        """
        if not q_series:
            return go.Figure()

        num_joints = q_series[0].shape[0]

        fig = make_subplots(
            rows=num_joints, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02
        )

        for i in range(num_joints):
            for j, q in enumerate(q_series):
                label = labels[j] if labels and j < len(labels) else f"Sim {j+1}"
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=q[i, :],
                        mode="lines",
                        name=label,
                        legendgroup=label,
                        showlegend=(i == 0)
                    ),
                    row=i + 1, col=1
                )
            
            if upper_bounds is not None and i < len(upper_bounds):
                fig.add_trace(go.Scatter(
                    x=t, y=np.atleast_1d(np.full_like(t, upper_bounds[i])),
                    mode='lines',
                    line=dict(color='red', dash='dot', width=1.5),
                    name='Upper Bound',
                    legendgroup='bounds',
                    showlegend=(i == 0)
                ), row=i + 1, col=1)

            if lower_bounds is not None and i < len(lower_bounds):
                fig.add_trace(go.Scatter(
                    x=t, y=np.atleast_1d(np.full_like(t, lower_bounds[i])),
                    mode='lines',
                    line=dict(color='red', dash='dot', width=1.5),
                    name='Lower Bound',
                    legendgroup='bounds',
                    showlegend=(i == 0)
                ), row=i + 1, col=1)

            fig.update_yaxes(title_text=f"${name}_{{{i+1}}}$ [{unit}]", row=i + 1, col=1)

        fig.update_xaxes(title_text="$t \\ [\\text{s}]$", row=num_joints, col=1)
        fig.update_layout(
            title=title,
            template=self.template,
            height=100 * num_joints + 60,
            margin=dict(l=50, r=20, t=40, b=40),
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
        lower_bound: float = None,
        upper_bound: float = None,
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
            lower_bound: Optional lower boundary line (scalar)
            upper_bound: Optional upper boundary line (scalar)
        """
        fig = go.Figure()
        x = np.atleast_1d(np.squeeze(x))

        # Plot main series
        for i, y in enumerate(y_series):
            y = np.atleast_1d(np.squeeze(y))
            label = labels[i] if labels and i < len(labels) else f"$S_{{{i+1}}}$"
            
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=label,
                )
            )

        # Add boundary lines if specified
        if lower_bound is not None:
            fig.add_trace(go.Scatter(
                x=x, 
                y=np.atleast_1d(np.full_like(x, lower_bound)),
                mode='lines',
                line=dict(color='red', dash='dot', width=1.5),
                name='Lower Bound',
            ))
        
        if upper_bound is not None:
            fig.add_trace(go.Scatter(
                x=x, 
                y=np.atleast_1d(np.full_like(x, upper_bound)),
                mode='lines',
                line=dict(color='red', dash='dot', width=1.5),
                name='Upper Bound',
            ))

        fig.update_layout(
            title=title,
            xaxis=dict(
                title=xlabel,
                type="log" if xlog else None,
            ),
            yaxis=dict(
                title=ylabel,
                type="log" if ylog else None,
            ),
            template=self.template,
            height=280,
            margin=dict(l=50, r=120, t=40, b=40),
        )

        return fig

    def bar_plot(
        self,
        values: np.ndarray,
        labels: list,
        xlabel: str = "Category",
        ylabel: str = "Value",
        title: str = "Bar Plot",
        ylog: bool = False,
    ):
        """Create a simple bar plot."""
        fig = go.Figure([go.Bar(x=labels, y=values)])
        fig.update_layout(
            title=title,
            xaxis=dict(title=xlabel),
            yaxis=dict(title=ylabel, type="log" if ylog else None),
            template=self.template,
            height=350,
            margin=dict(l=50, r=20, t=40, b=40),
        )
        return fig

    def grouped_bar_plot(
        self,
        data: dict,
        group_labels: list,
        xlabel: str = "Group",
        ylabel: str = "Value",
        title: str = "Grouped Bar Plot",
        ylog: bool = False,
    ):
        """Create a grouped bar plot."""
        fig = go.Figure()

        for series_name, values in data.items():
            fig.add_trace(go.Bar(x=group_labels, y=values, name=series_name))

        fig.update_layout(
            barmode='group',
            title=title,
            xaxis=dict(title=xlabel),
            yaxis=dict(title=ylabel, type="log" if ylog else None),
            template=self.template,
            height=350,
            margin=dict(l=50, r=20, t=40, b=40),
        )
        return fig

    def box_plot(
        self,
        data: dict,
        xlabel: str = "Category",
        ylabel: str = "Value",
        title: str = "Box Plot",
        ylog: bool = False,
        show_points: bool = True,
    ):
        """
        Create a box plot for multiple series with optional point overlay.
        
        Args:
            data: Dictionary where keys are series names and values are lists/arrays of data.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            title: Plot title.
            ylog: Use logarithmic scale for y-axis.
            show_points: Show individual data points as strip plot.
        """
        fig = go.Figure()

        for name, y_data in data.items():
            # Add box plot
            fig.add_trace(go.Box(
                y=y_data,
                name=name,
                boxpoints=False,  # Turn off default points
                whiskerwidth=0.2,
                line_width=1,
                showlegend=False,
            ))
            
            # Add strip plot alongside
            if show_points:
                fig.add_trace(go.Box(
                    y=y_data,
                    name=name,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.5,  # Position points to the left of box
                    marker=dict(
                        size=4,
                        opacity=0.5,
                        color='rgba(0,0,0,0.3)'
                    ),
                    line=dict(width=0),  # No box outline
                    fillcolor='rgba(255,255,255,0)',  # Transparent box
                    showlegend=False,
                ))

        fig.update_layout(
            title=title,
            yaxis=dict(
                autorange=True,
                showgrid=True,
                zeroline=True,
                title=ylabel,
                type="log" if ylog else "linear"
            ),
            xaxis=dict(title=xlabel),
            template=self.template,
            showlegend=False,
            height=400,
            margin=dict(l=50, r=20, t=40, b=40),
        )
        return fig

    def stacked_bar_plot(
        self,
        data: dict,
        group_labels: list,
        xlabel: str = "Group",
        ylabel: str = "Value",
        title: str = "Stacked Bar Plot",
        ylog: bool = False,
    ):
        """Create a stacked bar plot."""
        fig = go.Figure()

        for series_name, values in data.items():
            fig.add_trace(go.Bar(x=group_labels, y=values, name=series_name))

        fig.update_layout(
            barmode='stack',
            title=title,
            xaxis=dict(title=xlabel),
            yaxis=dict(title=ylabel, type="log" if ylog else None),
            template=self.template,
            height=350,
            margin=dict(l=50, r=20, t=40, b=40),
        )
        return fig
    
    # def grid_search_heatmap(
    #     self,
    #     sims: list,
    #     param_x: str,
    #     param_y: str,
    #     metric: str = 'rmse_e1',
    #     xlabel: str = None,
    #     ylabel: str = None,
    #     title: str = "Grid Search Heatmap",
    #     colorbar_title: str = None,
    # ):
    #     """
    #     Create a heatmap visualization of grid search results.
        
    #     Args:
    #         sims: List of simulator objects from grid search
    #         param_x: Parameter name for x-axis (e.g., 'prediction_horizon')
    #         param_y: Parameter name for y-axis (e.g., 'dt')
    #         metric: Metric to visualize (e.g., 'rmse_e1', 'itse_e1', 'avg_mpc_time')
    #                 Can be any attribute available on the simulator object.
    #         xlabel: X-axis label (defaults to param_x)
    #         ylabel: Y-axis label (defaults to param_y)
    #         title: Plot title
    #         colorbar_title: Title for colorbar (defaults to metric name)
        
    #     Returns:
    #         Plotly Figure object
            
    #     Example:
    #         fig = plotter.grid_search_heatmap(
    #             sims=sims,
    #             param_x='prediction_horizon',
    #             param_y='dt',
    #             metric='rmse_e1',
    #             xlabel='Prediction Horizon',
    #             ylabel='Time Step [s]',
    #             title='$e_1$ RMSE vs. MPC Parameters'
    #         )
    #     """
    #     # Extract unique parameter values
    #     x_values = []
    #     y_values = []
    #     z_values = []
        
    #     for sim in sims:
    #         # Get parameter values from simulation
    #         x_val = sim.get_results().get(param_x)
    #         y_val = sim.get_results().get(param_y)
            
    #         # If not in results, try to get from simulator attributes
    #         if x_val is None:
    #             x_val = getattr(sim, param_x, None)
    #         if y_val is None:
    #             y_val = getattr(sim, param_y, None)
                
    #         # Get metric value
    #         if hasattr(sim, metric):
    #             z_val = getattr(sim, metric)
    #         else:
    #             # Try to get from summary stats
    #             stats = sim.get_summary_stats()
    #             z_val = stats.get(metric)
            
    #         if x_val is not None and y_val is not None and z_val is not None:
    #             x_values.append(x_val)
    #             y_values.append(y_val)
    #             z_values.append(z_val)
        
    #     # Get unique sorted values for axes
    #     x_unique = sorted(list(set(x_values)))
    #     y_unique = sorted(list(set(y_values)))
        
    #     # Create 2D grid for heatmap
    #     z_grid = np.full((len(y_unique), len(x_unique)), np.nan)
        
    #     for x_val, y_val, z_val in zip(x_values, y_values, z_values):
    #         i = y_unique.index(y_val)
    #         j = x_unique.index(x_val)
    #         z_grid[i, j] = z_val
        
    #     # Create heatmap
    #     fig = go.Figure(data=go.Heatmap(
    #         z=z_grid,
    #         x=x_unique,
    #         y=y_unique,
    #         colorscale='Viridis',
    #         colorbar=dict(
    #             title=colorbar_title if colorbar_title else metric,
    #         ),
    #         hovertemplate=(
    #             f'{xlabel or param_x}: %{{x}}<br>' +
    #             f'{ylabel or param_y}: %{{y}}<br>' +
    #             f'{colorbar_title or metric}: %{{z:.4f}}<br>' +
    #             '<extra></extra>'
    #         ),
    #     ))
        
    #     # Add text annotations with values
    #     for i, y_val in enumerate(y_unique):
    #         for j, x_val in enumerate(x_unique):
    #             if not np.isnan(z_grid[i, j]):
    #                 fig.add_annotation(
    #                     x=x_val,
    #                     y=y_val,
    #                     text=f'{z_grid[i, j]:.4f}',
    #                     showarrow=False,
    #                     font=dict(color='white', size=10),
    #                 )
        
    #     fig.update_layout(
    #         title=title,
    #         xaxis=dict(title=xlabel if xlabel else param_x),
    #         yaxis=dict(title=ylabel if ylabel else param_y),
    #         template=self.template,
    #         height=400,
    #         margin=dict(l=50, r=20, t=40, b=40),
    #     )
        
    #     return fig

    def gen_html_report(
        self,
        task_figs: list = None,
        solver_figs: list = None,
        video_folder: str = None,
        title: str = "Analysis Report",
        filename: str = "report.html",
    ):
        """Generate HTML report with Video, Task Performance, and Solver Performance sections."""
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