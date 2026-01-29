import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, template: str = "ggplot2"):
        self.template = template

    def joints(
        self,
        t: np.ndarray,
        *q_series: np.ndarray,
        labels: list = None,
        title: str = "Joint Velocities",
        name: str = "q",
        unit: str = "rad/s",
        lower_bounds: np.ndarray = None,
        upper_bounds: np.ndarray = None,
    ):
        if not q_series:
            return go.Figure().update_layout(template=self.template)

        t = np.atleast_1d(np.squeeze(t))
        num_joints = q_series[0].shape[0]

        # --- Build figure first (template defines the default colorway used by generic_plot) ---
        fig = make_subplots(
            rows=num_joints, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02
        )
        fig.update_layout(template=self.template)

        # --- Use SAME palette as generic_plot (i.e., template colorway) ---
        colorway = None
        # 1) from template if available
        if fig.layout.template is not None and fig.layout.template.layout is not None:
            colorway = fig.layout.template.layout.colorway
        # 2) fallback
        if not colorway:
            colorway = px.colors.qualitative.Plotly

        if labels is None:
            labels = [f"Sim {k+1}" for k in range(len(q_series))]

        color_map = {lab: colorway[k % len(colorway)] for k, lab in enumerate(labels)}

        for i in range(num_joints):
            # --- main series (forced color per label => no swapping across subplots) ---
            for j, q in enumerate(q_series):
                label = labels[j] if j < len(labels) else f"Sim {j+1}"
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

            # --- bounds (added AFTER series so they stay on top) ---
            if upper_bounds is not None and i < len(upper_bounds) and upper_bounds[i] is not None:
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=np.full_like(t, upper_bounds[i], dtype=float),
                        mode="lines",
                        name="Upper Bound",
                        legendgroup="bounds",
                        showlegend=(i == 0),
                        line=dict(color="red", dash="dot", width=2.5),
                    ),
                    row=i + 1, col=1
                )

            if lower_bounds is not None and i < len(lower_bounds) and lower_bounds[i] is not None:
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=np.full_like(t, lower_bounds[i], dtype=float),
                        mode="lines",
                        name="Lower Bound",
                        legendgroup="bounds",
                        showlegend=(i == 0),
                        line=dict(color="red", dash="dot", width=2.5),
                    ),
                    row=i + 1, col=1
                )

            fig.update_yaxes(title_text=f"${name}_{{{i+1}}}$ [{unit}]", row=i + 1, col=1)

        fig.update_xaxes(title_text="$t \\ [\\text{s}]$", row=num_joints, col=1)
        fig.update_layout(
            title=title,
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
        fig.update_layout(template=self.template)   # <<< FONDAMENTALE
        x = np.atleast_1d(np.squeeze(x))

        # --- palette coerente col template ---
        # --- palette coerente col template ---
        colorway = None
        if fig.layout.template is not None and fig.layout.template.layout is not None:
            colorway = fig.layout.template.layout.colorway

        # estendi se troppo corta
        if colorway and len(colorway) == 5:
            colorway = list(colorway) + ["#FF9F00"]

        # labels robusto (None o [])
        if not labels:
            labels = [f"$S_{{{i+1}}}$" for i in range(len(y_series))]


        def get_color(i):
            return colorway[i % len(colorway)]


        # Plot main series (colore fissato per label)
        for i, y in enumerate(y_series):
            y = np.atleast_1d(np.squeeze(y))
            label = labels[i] if i < len(labels) else f"$S_{{{i+1}}}$"

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    line=dict(
                        width=2.5,
                        color=get_color(i),   # <<< colore per indice
                    ),
                )
            )


        # Add boundary lines if specified
        if lower_bound is not None:
            fig.add_trace(go.Scatter(
                x=x, 
                y=np.atleast_1d(np.full_like(x, lower_bound)),
                mode='lines',
                line=dict(color='red', dash='dot', width=4.0),
                name='Lower Bound',
            ))
        
        if upper_bound is not None:
            fig.add_trace(go.Scatter(
                x=x, 
                y=np.atleast_1d(np.full_like(x, upper_bound)),
                mode='lines',
                line=dict(color='red', dash='dot', width=3.0),
                name='MPC sampling time',
            ))

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=50)   
            ),
            xaxis=dict(
                title=dict(
                    text=xlabel,
                    font=dict(size=20)  
                ),
                type="log" if xlog else None,
            ),
            yaxis=dict(
                title=dict(
                    text=ylabel,
                    font=dict(size=20)   
                ),
                type="log" if ylog else None,
            ),
            template=self.template,
            colorway=colorwayy,
            height=800,
            margin=dict(l=240, r=240, t=60, b=60),
            legend=dict(
                font=dict(size=16),
                itemsizing="constant",
                itemwidth=30,
                xanchor="left",
                yanchor="middle",
                x=1.02,       
                y=0.5,         
                bgcolor="rgba(255,255,255,0.0)", 
                borderwidth=0,
            )





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
        point_labels: list = None,
        point_colors: list = None,
        lower_bound: float = None,
        upper_bound: float = None,
    ):
        """
        Create a box plot for multiple series with optional colored point overlay.
        
        Args:
            data: Dictionary where keys are series names and values are lists/arrays of data.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            title: Plot title.
            ylog: Use logarithmic scale for y-axis.
            show_points: Show individual data points as strip plot.
            point_labels: Labels for each point (used in legend). Same length as each data array.
            point_colors: Colors for each point. Same length as each data array.
            lower_bound: Optional lower boundary line (scalar).
            upper_bound: Optional upper boundary line (scalar).
        """
        fig = go.Figure()
        
        # Color palette for scatter points
        default_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        for name, y_data in data.items():
            y_data = np.atleast_1d(y_data)
            
            # Add box plot
            fig.add_trace(go.Box(
                y=y_data,
                name=name,
                boxpoints=False,  # Turn off default points
                whiskerwidth=0.2,
                line_width=1,
                showlegend=False,
            ))
            
            # Add scatter points with colors
            if show_points:
                if point_labels is not None and point_colors is not None:
                    # Group points by label for legend
                    unique_labels = list(dict.fromkeys(point_labels))  # Preserve order
                    for i, label in enumerate(unique_labels):
                        mask = [pl == label for pl in point_labels]
                        y_subset = [y for y, m in zip(y_data, mask) if m]
                        if y_subset:
                            # Add jitter manually
                            x_jitter = [name] * len(y_subset)
                            fig.add_trace(go.Scatter(
                                x=x_jitter,
                                y=y_subset,
                                mode='markers',
                                name=label,
                                legendgroup=label,
                                showlegend=(name == list(data.keys())[0]),  # Show legend only for first box
                                marker=dict(
                                    size=8,
                                    color=point_colors[i] if i < len(point_colors) else default_colors[i % len(default_colors)],
                                    opacity=0.7,
                                ),
                            ))
                else:
                    # Default behavior: all points same color
                    fig.add_trace(go.Box(
                        y=y_data,
                        name=name,
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.5,
                        marker=dict(
                            size=4,
                            opacity=0.5,
                            color='rgba(0,0,0,0.3)'
                        ),
                        line=dict(width=0),
                        fillcolor='rgba(255,255,255,0)',
                        showlegend=False,
                    ))

        # Add boundary lines if specified
        x_categories = list(data.keys())
        if lower_bound is not None:
            fig.add_trace(go.Scatter(
                x=x_categories,
                y=[lower_bound] * len(x_categories),
                mode='lines',
                line=dict(color='red', dash='dot', width=1.5),
                name='Lower Bound',
                showlegend=True,
            ))
        
        if upper_bound is not None:
            fig.add_trace(go.Scatter(
                x=x_categories,
                y=[upper_bound] * len(x_categories),
                mode='lines',
                line=dict(color='red', dash='dot', width=1.5),
                name='Upper Bound',
                showlegend=True,
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
            showlegend=point_labels is not None,
            legend=dict(title="Surface Config") if point_labels is not None else None,
            height=450,
            margin=dict(l=50, r=120, t=40, b=40),
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


    def gen_html_report(
    self,
    task_figs: list = None,
    solver_figs: list = None,
    video_folder: str = None,
    title: str = "Analysis Report",
    filename: str = "report.html",
    output_dir: str = None,
    open_browser: bool = True,
    ):
        """Generate HTML report with Video, Task Performance, and Solver Performance sections.

        Args:
            task_figs: list of plotly figures
            solver_figs: list of plotly figures
            video_folder: folder containing .mp4 files (optional)
            title: report title
            filename: html filename (e.g. 'run_01.html')
            output_dir: output directory (default: <repo_root>/Reporting/HTML)
            open_browser: open report in browser
        Returns:
            Absolute path (string) to the generated HTML report.
        """

        # --- resolve output dir robustly (independent from cwd) ---
        repo_root = Path(__file__).resolve().parents[1]  # .../Reporting/plotter.py -> repo root
        out_dir = Path(output_dir) if output_dir is not None else (repo_root / "Reporting" / "HTML")
        out_dir.mkdir(parents=True, exist_ok=True)

        # If user passes a path in filename, respect it; otherwise put it under out_dir
        filename_path = Path(filename)
        report_path = filename_path if filename_path.is_absolute() or filename_path.parent != Path(".") else (out_dir / filename_path)

        sections_html = []
        plotly_included = False
        plot_counter = 0

        # --- Video section ---
        if video_folder:
            video_dir = Path(video_folder)
            if video_dir.is_dir():
                video_files = sorted([p for p in video_dir.iterdir() if p.suffix.lower() == ".mp4"])
                if video_files:
                    video_cells = []
                    for video_file in video_files:
                        video_name = video_file.stem
                        # Use absolute file:// path (works reliably locally)
                        video_uri = video_file.resolve().as_uri()
                        video_html = (
                            f"<h4>{video_name}</h4>"
                            f"<video controls loop muted playsinline>"
                            f"<source src='{video_uri}' type='video/mp4'>"
                            f"</video>"
                        )
                        video_cells.append(f"<div class='cell'>{video_html}</div>")

                    sections_html.append(f"""
                    <section>
                        <h2>Video</h2>
                        <div class="grid">
                            {''.join(video_cells)}
                        </div>
                    </section>""")

        # --- Task Performance section ---
        if task_figs:
            task_cells = []
            for fig in task_figs:
                div_id = f"plot-{plot_counter}"
                plot_counter += 1
                include_js = "cdn" if not plotly_included else False
                if include_js == "cdn":
                    plotly_included = True

                html = pio.to_html(
                    fig,
                    full_html=False,
                    include_plotlyjs=include_js,
                    include_mathjax=False,
                    div_id=div_id,
                    config={"responsive": True},
                )
                task_cells.append(f"<div class='cell'>{html}</div>")

            sections_html.append(f"""
            <section>
                <h2>Task Performance</h2>
                <div class="grid">
                    {''.join(task_cells)}
                </div>
            </section>""")

        # --- Solver Performance section ---
        if solver_figs:
            solver_cells = []
            for fig in solver_figs:
                div_id = f"plot-{plot_counter}"
                plot_counter += 1
                include_js = "cdn" if not plotly_included else False
                if include_js == "cdn":
                    plotly_included = True

                html = pio.to_html(
                    fig,
                    full_html=False,
                    include_plotlyjs=include_js,
                    include_mathjax=False,
                    div_id=div_id,
                    config={"responsive": True},
                )
                solver_cells.append(f"<div class='cell'>{html}</div>")

            sections_html.append(f"""
            <section>
                <h2>Solver Performance</h2>
                <div class="grid">
                    {''.join(solver_cells)}
                </div>
            </section>""")

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
        {''.join(sections_html)}
        <script>
            function resizePlots() {{
                document.querySelectorAll('.js-plotly-plot').forEach(p => Plotly.Plots.resize(p));
            }}
            window.addEventListener('resize', resizePlots);
            window.addEventListener('load', resizePlots);
        </script>
    </body>
    </html>"""

        report_path = report_path.resolve()
        report_path.write_text(html_content, encoding="utf-8")

        if open_browser:
            webbrowser.open(report_path.as_uri())

        return str(report_path)


    def fig6_computation_time_boxplot_mpl(
    self,
    sims_or_results,
    *,
    time_source: str = "mpc_time",    
    sample_time_s: float | None = None,
    title: str = "",
    figsize=(10, 4.2),
    box_width: float | None = None,    
    box_edge_color: str = "#3566A6",   
    box_face_color: str = "#3566A6",   
    median_color: str = "#234776",
    outlier_edge_color: str = "#234776",
    outlier_size: float = 6.5,
):
        """
        Paper-like Fig.6 (MATPLOTLIB): one box per horizon length (Ts*N),
        log-scale y, Tukey whiskers (whis=1.5), hollow outliers drawn by matplotlib.

        Boxes are uniformly spaced along x (categorical positions 1..K),
        while x-tick labels show the horizon length in seconds (rounded to 3 decimals).

        Returns:
            fig, ax (matplotlib)
        """
        # unwrap results from SimulationManager
        sims = []
        for item in sims_or_results:
            sims.append(item["simulator"] if isinstance(item, dict) and "simulator" in item else item)

        fig, ax = plt.subplots(figsize=figsize)

        if len(sims) == 0:
            return fig, ax

        if sample_time_s is None:
            sample_time_s = float(sims[0].dt)

        # collect data: one group per horizon length (seconds)
        horizons = []
        data = []
        for sim in sims:
            h_s = float(sim.dt * sim.prediction_horizon)
            y = np.asarray(sim.timings[time_source]).ravel().astype(float)
            y = np.maximum(y, 1e-12)  # avoid zeros for log-scale
            horizons.append(h_s)
            data.append(y)

        # sort by horizon
        order = np.argsort(horizons)
        horizons = [horizons[i] for i in order]
        data = [data[i] for i in order]

        # --- UNIFORM x spacing (categorical) ---
        k = len(horizons)
        xpos = np.arange(1, k + 1)

        # box width in categorical units
        if box_width is None:
            box_width = 0.55

        # --- props (ALL handled by matplotlib.boxplot) ---
        boxprops = dict(facecolor=box_face_color, edgecolor=box_edge_color, linewidth=1.2, alpha=0.25)
        whiskerprops = dict(color=box_edge_color, linewidth=1.2)
        capprops = dict(color=box_edge_color, linewidth=1.2)
        medianprops = dict(color=median_color, linewidth=1.4)

        # hollow, dark outliers (fliers)
        flierprops = dict(
            marker='o',
            markerfacecolor='none',         
            markeredgecolor=outlier_edge_color,
            markersize=outlier_size,
            linestyle='none',
            markeredgewidth=0.9,
            alpha=0.85
        )

        bp = ax.boxplot(
            data,
            positions=xpos,
            widths=box_width,
            patch_artist=True,
            showfliers=True,  
            whis=1.5,         
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops,
            flierprops=flierprops,
        )

        # dashed red sample-time line + legend
        ax.axhline(
            sample_time_s,
            color="red",
            linestyle="--",
            linewidth=2.5,
            label="MPC sample time"
        )

        # axes styling
        ax.set_yscale("log")
        ax.set_xlabel("Tested surface instances")
        ax.set_ylabel("MPC computatio time")
        if title:
            ax.set_title(title)

        ax.grid(True, which="major", axis="both", alpha=0.35)
        ax.grid(True, which="minor", axis="y", alpha=0.15)

        # x ticks: one tick per tested surface instance
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(i) for i in xpos])

        # legend: show only the dashed line (avoid box proxy in legend)
        
      
        ax.text(
            0.02, 0.92,                 
            "MPC sample time",
            transform=ax.transAxes,      
            color="red",
            fontsize=11,
            va="top",
            ha="left"
        )

        fig.tight_layout()
        return fig, ax


    def error_envelope(
        self,
        sims_or_results,
        *,
        error_key: str = "e4",          
        band: str = "std",             
        k_sigma: float = 1.0,           
        t_max: float | None = None,     
        show_individual: int = 0,       
        title: str = "",
        xlabel: str = "$t\\ [\\mathrm{s}]$",
        ylabel: str | None = None,
    ):
            """
            Plot mean error trajectory with variability band across multiple simulations.

            sims_or_results: list of Simulator or list of dicts with key "simulator"
            error_key: one of 'e1'..'e5'
            band:
            - "std": mean ± k_sigma * std
            - "minmax": [min, max]
            show_individual: plot up to N individual trajectories with low opacity
            """
            # unwrap results (same pattern as fig6_computation_time_boxplot_mpl)
            sims = []
            for item in sims_or_results:
                sims.append(item["simulator"] if isinstance(item, dict) and "simulator" in item else item)

            if len(sims) == 0:
                return go.Figure()

            # assume same dt; align lengths robustly (truncate to shortest)
            dt = float(sims[0].dt)
            e_list = []
            n_min = None

            for sim in sims:
                e = np.asarray(sim.errors[error_key]).ravel().astype(float) 
                n_min = len(e) if n_min is None else min(n_min, len(e))
                e_list.append(e)

            # truncate and stack -> shape (K, T)
            E = np.vstack([e[:n_min] for e in e_list])
            t = np.arange(n_min) * dt

            if t_max is not None:
                n_cut = int(np.floor(t_max / dt))
                n_cut = max(2, min(n_cut, n_min))
                t = t[:n_cut]
                E = E[:, :n_cut]

            mean = np.mean(E, axis=0)

            if band.lower() == "minmax":
                lo = np.min(E, axis=0)
                hi = np.max(E, axis=0)
                band_name = "min–max"
            elif band.lower() == "std":
                std = np.std(E, axis=0)
                lo = mean - k_sigma * std
                hi = mean + k_sigma * std
                band_name = f"$\\pm {k_sigma:.1f}\\sigma$"
            else:
                raise ValueError("band must be 'std' or 'minmax'")

            if ylabel is None:
                ylabel = f"${error_key}(t)$"

            fig = go.Figure()
            fig.update_layout(template=self.template)

            # (optional) individual trajectories in the background
            if show_individual and show_individual > 0:
                n_show = min(int(show_individual), E.shape[0])
                for i in range(n_show):
                    fig.add_trace(
                        go.Scatter(
                            x=t, y=E[i, :],
                            mode="lines",
                            line=dict(width=1.5),
                            opacity=0.25,
                            name=f"instance {i+1}",
                            showlegend=False,
                        )
                    )

            # band: lower then upper with fill
            fig.add_trace(
                go.Scatter(
                    x=t, y=lo,
                    mode="lines",
                    line=dict(width=0),
                    name=band_name,
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=t, y=hi,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(53,102,166,0.18)",  # blu trasparente
                    name=band_name,
                )
            )

            # mean line on top
            fig.add_trace(
                go.Scatter(
                    x=t, y=mean,
                    mode="lines",
                    line=dict(width=3.0),
                    name="mean",
                )
            )

            fig.update_layout(
                title=dict(text=title, font=dict(size=40)) if title else None,
                xaxis=dict(title=dict(text=xlabel, font=dict(size=20))),
                yaxis=dict(title=dict(text=ylabel, font=dict(size=20))),
                height=650,
                margin=dict(l=120, r=120, t=80, b=80),
                legend=dict(
                    font=dict(size=16),
                    xanchor="right",
                    yanchor="top",
                    x=0.98,
                    y=0.98,
                    bgcolor="rgba(255,255,255,0.0)",
                    borderwidth=0,
                )
            )

            return fig




    def fig6_computation_time_boxplot_mpl(
    self,
    sims_or_results,
    *,
    time_source: str = "mpc_time",    
    sample_time_s: float | None = None,
    title: str = "",
    figsize=(10, 4.2),
    box_width: float | None = None,    
    box_edge_color: str = "#3566A6",   
    box_face_color: str = "#3566A6",   
    median_color: str = "#234776",
    outlier_edge_color: str = "#234776",
    outlier_size: float = 6.5,
):
        """
        Paper-like Fig.6 (MATPLOTLIB): one box per horizon length (Ts*N),
        log-scale y, Tukey whiskers (whis=1.5), hollow outliers drawn by matplotlib.

        Boxes are uniformly spaced along x (categorical positions 1..K),
        while x-tick labels show the horizon length in seconds (rounded to 3 decimals).

        Returns:
            fig, ax (matplotlib)
        """
        # unwrap results from SimulationManager
        sims = []
        for item in sims_or_results:
            sims.append(item["simulator"] if isinstance(item, dict) and "simulator" in item else item)

        fig, ax = plt.subplots(figsize=figsize)

        if len(sims) == 0:
            return fig, ax

        if sample_time_s is None:
            sample_time_s = float(sims[0].dt)

        # collect data: one group per horizon length (seconds)
        horizons = []
        data = []
        for sim in sims:
            h_s = float(sim.dt * sim.prediction_horizon)
            y = np.asarray(sim.timings[time_source]).ravel().astype(float)
            y = np.maximum(y, 1e-12)  # avoid zeros for log-scale
            horizons.append(h_s)
            data.append(y)

        # sort by horizon
        order = np.argsort(horizons)
        horizons = [horizons[i] for i in order]
        data = [data[i] for i in order]

        # --- UNIFORM x spacing (categorical) ---
        k = len(horizons)
        xpos = np.arange(1, k + 1)

        # box width in categorical units
        if box_width is None:
            box_width = 0.55

        # --- props (ALL handled by matplotlib.boxplot) ---
        boxprops = dict(facecolor=box_face_color, edgecolor=box_edge_color, linewidth=1.2, alpha=0.25)
        whiskerprops = dict(color=box_edge_color, linewidth=1.2)
        capprops = dict(color=box_edge_color, linewidth=1.2)
        medianprops = dict(color=median_color, linewidth=1.4)

        # hollow, dark outliers (fliers)
        flierprops = dict(
            marker='o',
            markerfacecolor='none',         
            markeredgecolor=outlier_edge_color,
            markersize=outlier_size,
            linestyle='none',
            markeredgewidth=0.9,
            alpha=0.85
        )

        bp = ax.boxplot(
            data,
            positions=xpos,
            widths=box_width,
            patch_artist=True,
            showfliers=True,  
            whis=1.5,         
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops,
            flierprops=flierprops,
        )

        # dashed red sample-time line + legend
        ax.axhline(
            sample_time_s,
            color="red",
            linestyle="--",
            linewidth=2.5,
            label="MPC sample time"
        )

        # axes styling
        ax.set_yscale("log")
        ax.set_xlabel("Tested surface instances")
        ax.set_ylabel("MPC computatio time")
        if title:
            ax.set_title(title)

        ax.grid(True, which="major", axis="both", alpha=0.35)
        ax.grid(True, which="minor", axis="y", alpha=0.15)

        # x ticks: one tick per tested surface instance
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(i) for i in xpos])

        # legend: show only the dashed line (avoid box proxy in legend)
        
      
        ax.text(
            0.02, 0.92,                 
            "MPC sample time",
            transform=ax.transAxes,      
            color="red",
            fontsize=11,
            va="top",
            ha="left"
        )

        fig.tight_layout()
        return fig, ax


    def error_envelope(
        self,
        sims_or_results,
        *,
        error_key: str = "e4",          
        band: str = "std",             
        k_sigma: float = 1.0,           
        t_max: float | None = None,     
        show_individual: int = 0,       
        title: str = "",
        xlabel: str = "$t\\ [\\mathrm{s}]$",
        ylabel: str | None = None,
    ):
            """
            Plot mean error trajectory with variability band across multiple simulations.

            sims_or_results: list of Simulator or list of dicts with key "simulator"
            error_key: one of 'e1'..'e5'
            band:
            - "std": mean ± k_sigma * std
            - "minmax": [min, max]
            show_individual: plot up to N individual trajectories with low opacity
            """
            # unwrap results (same pattern as fig6_computation_time_boxplot_mpl)
            sims = []
            for item in sims_or_results:
                sims.append(item["simulator"] if isinstance(item, dict) and "simulator" in item else item)

            if len(sims) == 0:
                return go.Figure()

            # assume same dt; align lengths robustly (truncate to shortest)
            dt = float(sims[0].dt)
            e_list = []
            n_min = None

            for sim in sims:
                e = np.asarray(sim.errors[error_key]).ravel().astype(float) 
                n_min = len(e) if n_min is None else min(n_min, len(e))
                e_list.append(e)

            # truncate and stack -> shape (K, T)
            E = np.vstack([e[:n_min] for e in e_list])
            t = np.arange(n_min) * dt

            if t_max is not None:
                n_cut = int(np.floor(t_max / dt))
                n_cut = max(2, min(n_cut, n_min))
                t = t[:n_cut]
                E = E[:, :n_cut]

            mean = np.mean(E, axis=0)

            if band.lower() == "minmax":
                lo = np.min(E, axis=0)
                hi = np.max(E, axis=0)
                band_name = "min–max"
            elif band.lower() == "std":
                std = np.std(E, axis=0)
                lo = mean - k_sigma * std
                hi = mean + k_sigma * std
                band_name = f"$\\pm {k_sigma:.1f}\\sigma$"
            else:
                raise ValueError("band must be 'std' or 'minmax'")

            if ylabel is None:
                ylabel = f"${error_key}(t)$"

            fig = go.Figure()
            fig.update_layout(template=self.template)

            # (optional) individual trajectories in the background
            if show_individual and show_individual > 0:
                n_show = min(int(show_individual), E.shape[0])
                for i in range(n_show):
                    fig.add_trace(
                        go.Scatter(
                            x=t, y=E[i, :],
                            mode="lines",
                            line=dict(width=1.5),
                            opacity=0.25,
                            name=f"instance {i+1}",
                            showlegend=False,
                        )
                    )

            # band: lower then upper with fill
            fig.add_trace(
                go.Scatter(
                    x=t, y=lo,
                    mode="lines",
                    line=dict(width=0),
                    name=band_name,
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=t, y=hi,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(53,102,166,0.18)",  # blu trasparente
                    name=band_name,
                )
            )

            # mean line on top
            fig.add_trace(
                go.Scatter(
                    x=t, y=mean,
                    mode="lines",
                    line=dict(width=3.0),
                    name="mean",
                )
            )

            fig.update_layout(
                title=dict(text=title, font=dict(size=40)) if title else None,
                xaxis=dict(title=dict(text=xlabel, font=dict(size=20))),
                yaxis=dict(title=dict(text=ylabel, font=dict(size=20))),
                height=650,
                margin=dict(l=120, r=120, t=80, b=80),
                legend=dict(
                    font=dict(size=16),
                    xanchor="right",
                    yanchor="top",
                    x=0.98,
                    y=0.98,
                    bgcolor="rgba(255,255,255,0.0)",
                    borderwidth=0,
                )
            )

            return fig




    def show(self, fig: go.Figure, renderer: str = None):
        """
        Show figure with LaTeX/MathJax support.
        
        Args:
            fig: Plotly figure to display
            renderer: Optional renderer override. Use "browser" for external browser,
                     None for auto-detect (works in Jupyter notebooks).
        """
        if renderer == "browser":
            fig.show(renderer="browser", config={'mathjax': 'cdn'})
        else:
            # For Jupyter notebooks: use notebook renderer with MathJax
            fig.show(config={'mathjax': 'cdn'})

    def export_svg(
        self,
        fig: go.Figure = None,
        figs: dict = None,
        filename: str = "plot.svg",
        output_dir: str = None,
        width: int = None,
        height: int = None,
        scale: float = 1.0,
    ):
        """
        Export figure(s) to SVG format.
        
        Args:
            fig: Single Plotly figure to export (use this OR figs, not both)
            figs: Dictionary of {name: figure} for batch export
            filename: Output filename (used when exporting single fig)
            output_dir: Output directory (default: current working directory)
            width: Image width in pixels (default: uses figure's width)
            height: Image height in pixels (default: uses figure's height)
            scale: Scale factor for the image (default: 1.0)
        
        Returns:
            Path to exported file (single) or list of paths (batch)
        
        Example:
            # Single figure
            plotter.export_svg(fig_e1, filename="error_e1.svg")
            
            # Batch export
            plotter.export_svg(figs={
                "error_e1": fig_e1,
                "error_e2": fig_e2,
                "sqp_iters": fig_sqp,
            }, output_dir="exports/")
        
        Note:
            Requires kaleido package: pip install kaleido
        """
        from pathlib import Path
        
        out_dir = Path(output_dir) if output_dir else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        
        export_kwargs = {"format": "svg", "scale": scale}
        if width is not None:
            export_kwargs["width"] = width
        if height is not None:
            export_kwargs["height"] = height
        
        if fig is not None:
            # Single figure export
            out_path = out_dir / filename
            fig.write_image(str(out_path), **export_kwargs)
            return str(out_path.resolve())
        
        elif figs is not None:
            # Batch export
            paths = []
            for name, figure in figs.items():
                out_path = out_dir / f"{name}.svg"
                figure.write_image(str(out_path), **export_kwargs)
                paths.append(str(out_path.resolve()))
            return paths
        
        else:
            raise ValueError("Must provide either 'fig' or 'figs' argument")
    
    @staticmethod
    def enable_latex_in_notebook():
        """
        Call this at the start of a Jupyter notebook to enable LaTeX rendering.
        
        Example:
            from plotter import Plotter
            Plotter.enable_latex_in_notebook()
        """
        from IPython.display import display, HTML
        display(HTML(
            '<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script>'
        ))