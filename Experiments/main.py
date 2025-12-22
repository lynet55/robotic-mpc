from Experiments.simulator import SimulationManager
from Reporting.plotter import Plotter

import numpy as np

plotter = Plotter(template="plotly_white")

# Define joint limits for plotting boundaries
JOINT_POSITION_LIMITS = np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
JOINT_VELOCITY_LIMITS = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]) # UR5e limits
JOINT_ACCELERATION_LIMITS = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]) # Example limitse parameters shared by all simulations

BASE_PARAMS = {
    'dt': 0.0005,
    'simulation_time': 2,
    'surface_limits': ((-2, 2), (-2, 2)),
    'surface_origin': np.array([0.0, 0.0, 0.0]),
    'surface_orientation_rpy': np.array([0.0, 0.0, 0.0]),
    #surface function coefficients for a paraboloid
    'qdot_0': np.array([2, 2, 2, 2, 2, 2]),
    'q_0': np.array([np.pi/3, -np.pi/3, np.pi/4, -np.pi/2, -np.pi/2, 0.0]),
    'wcv': np.array([228.9, 262.09, 517.3, 747.44, 429.9, 1547.76]),
    'q_min': np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi], dtype=float),
    'q_max': np.array([+2*np.pi, +2*np.pi, +np.pi, +2*np.pi, +2*np.pi, +2*np.pi], dtype=float),
    'qdot_min': np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi], dtype=float),
    'qdot_max': np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi], dtype=float),
    'scene': True,
    'prediction_horizon': 200,  # Default value
    'solver_options': [{ 
        'nlp_solver_type': 'SQP',
        'qp_solver': 'PARTIAL_CONDENSING_HPIPM'
    }]
}

# ============================================================================
# SIMULATION QUEUE - Define all simulations to run
# ============================================================================

# 1. Initialize the manager with base parameters
manager = SimulationManager(BASE_PARAMS)

# Example 1: Single parameter sweep
# manager.sweep_parameter('prediction_horizon', [10, 50, 100, 200, 300], name_template='H={}')
# surface coeffs random gen sweep, plot variance in errors. mean error. 

# Example 2: Grid search over multiple parameters
manager.grid_search(
    {
        'prediction_horizon': [100],
        'dt': [0.001],
        'solver_options': [
            {
                'nlp_solver_type': 'SQP',
                'qp_solver': 'PARTIAL_CONDENSING_HPIPM',
            }
        ],
    },
    name_template=lambda p, _: (
        f"H={p['prediction_horizon']}_dt={p['dt']}_"
        f"{p['solver_options']['nlp_solver_type']}_"
        f"{p['solver_options']['qp_solver']}"
    )
)

# name_template=lambda p: f"H={p['prediction_horizon']}_dt={p['dt']}_{p['solver_options']['nlp_solver_type']}")

# Example 3: Manual definition (original approach)
# manager.add_manual(name='H=200', params={'prediction_horizon': 200})
# manager.add_manual(name='H=10', params={'prediction_horizon': 10})
# manager.add_manual(name='H=50', params={'prediction_horizon': 50})

# 2. Run all queued simulations
results = manager.run_all()
sims = [r['simulator'] for r in results]


# ============================================================================
# PLOTTING - Automatically plot all simulations
# ============================================================================

# Use first simulation for time axis and single-sim plots
ref_sim = sims[0]
t = ref_sim.get_data()['time']
summaries = [s.get_summary() for s in sims]


# Grouped bar plot for RMSE comparison
rmse_data = {
    "$e_1$ (surface)": [sm['rmse_e1'] for sm in summaries],
    "$e_2$ (x-pos)":   [sm['rmse_e2'] for sm in summaries],
    "$e_3$ (orient)":  [sm['rmse_e3'] for sm in summaries],
    "$e_4$ (y-vel)":   [sm['rmse_e4'] for sm in summaries],
}


fig_rmse = plotter.grouped_bar_plot(
    data=rmse_data,
    group_labels=[s.name for s in sims],
    xlabel="Simulation",
    ylabel="RMSE",
    title="Constraint RMSE Comparison"
)

# Grouped bar plot for ITSE comparison
itse_data = {
    "$e_1$ (surface)": [sm['itse_e1'] for sm in summaries],
    "$e_2$ (x-pos)":   [sm['itse_e2'] for sm in summaries],
    "$e_3$ (orient)":  [sm['itse_e3'] for sm in summaries],
    "$e_4$ (y-vel)":   [sm['itse_e4'] for sm in summaries],
}
fig_itse = plotter.grouped_bar_plot(
    data=itse_data,
    group_labels=[s.name for s in sims],
    xlabel="Simulation",
    ylabel="ITSE",
    title="Constraint ITSE Comparison"
)
# Bar chart for total computation time
fig_total_time = plotter.bar_plot(
    values=[s.get_summary()['total_computation_time'] for s in sims],
    labels=[s.name for s in sims],
    xlabel="Simulation",
    ylabel="Total Time [s]",
    title="Total Computation Time (MPC + Integration)"
)


# Joint plots comparing all simulations
fig_joints_q = plotter.joints(
    t, *[s.get_data()['q'] for s in sims], 
    labels=[s.name for s in sims],
    title="Joint Angles", name="q", unit="rad",
    lower_bounds=-JOINT_POSITION_LIMITS,
    upper_bounds=JOINT_POSITION_LIMITS
)
fig_joints_qdot = plotter.joints(
    t, *[s.get_data()['qdot'] for s in sims], 
    labels=[s.name for s in sims],
    title="Joint Velocities", name="\\dot{q}", unit="rad/s",
    lower_bounds=-JOINT_VELOCITY_LIMITS,
    upper_bounds=JOINT_VELOCITY_LIMITS
)
fig_joints_qdotdot = plotter.joints(
    t, *[s.get_data()['u'] for s in sims], 
    labels=[s.name for s in sims],
    title="Joint Accelerations (Control Input)", name="\\ddot{q}", unit="rad/s²",
    lower_bounds=-JOINT_ACCELERATION_LIMITS,
    upper_bounds=JOINT_ACCELERATION_LIMITS
)

# Control Input Plot for reference sim
u_ref = ref_sim.get_data()['u']
fig_u = plotter.generic_plot(
    t,
    u_ref[0], u_ref[1], u_ref[2], u_ref[3], u_ref[4], u_ref[5],
    xlabel="$t \\ [\\text{s}]$",
    ylabel="$u$ [rad/s]",
    title=f"Control Inputs - {ref_sim.name}",
    labels=["$\dot{q}_{1}$", "$\dot{q}_{2}$", "$\dot{q}_{3}$", 
            "$\dot{q}_{4}$", "$\dot{q}_{5}$", "$\dot{q}_{6}$"]
)

# ============================================================================
# COMPARATIVE PLOTS - All simulations overlaid
# ============================================================================

# Constraint error plots
fig_e1 = plotter.generic_plot(
    t, 
    *[s.errors['e1'] for s in sims],
    xlabel="$t \\ [\\text{s}]$", 
    ylabel="$e_1 \\ [\\text{m}]$", 
    title="$e_1 = s(x,y) - z$", 
    labels=[s.name for s in sims]
)


fig_e2 = plotter.generic_plot(
    t, 
    *[s.errors['e2'] for s in sims],
    xlabel="$t \\ [\\text{s}]$", 
    ylabel="$e_2 \\ [\\text{m}]$", 
    title="$e_2 = p_{x,\\text{ref}} - p_x$", 
    labels=[s.name for s in sims]
)

fig_e3 = plotter.generic_plot(
    t, 
    *[s.errors['e3'] for s in sims],
    xlabel="$t \\ [\\text{s}]$", 
    ylabel="$e_3$", 
    title="$e_3 = n^T R_{\\text{ee}} v_{\\text{ee}}$", 
    labels=[s.name for s in sims]
)

fig_e4 = plotter.generic_plot(
    t, 
    *[s.errors['e4'] for s in sims],
    xlabel="$t \\ [\\text{s}]$", 
    ylabel="$e_4 \\ [\\text{m/s}]$", 
    title="$e_4 = v_{\\text{ref}} - v_{\\text{ee}}$", 
    labels=[s.name for s in sims]
)

# Control Input Comparison - Joint 1
fig_u1 = plotter.generic_plot(
    t, 
    *[s.get_data()['u'][0] for s in sims],
    xlabel="$t \\ [\\text{s}]$", 
    ylabel="$u_1 \\ [\\text{rad/s}]$", 
    title="Control Input - Joint 1", 
    labels=[s.name for s in sims]
)

# KKT Residuals
fig_kkt = plotter.generic_plot(
    t, 
    *[s.solver_stats['kkt_residuals'] for s in sims],
    ylog=True, 
    xlabel="$t \\ [\\text{s}]$", 
    ylabel="$\\text{KKT Residual (max)}$", 
    title="KKT Residuals over Simulation", 
    labels=[s.name for s in sims]
)

# Individual residual components for reference sim
res = ref_sim.get_analysis()
fig_res_components = plotter.generic_plot(
    t,
    res['res_stat'],
    res['res_eq'],
    res['res_ineq'],
    res['res_comp'],
    ylog=True,
    xlabel="$t \\ [\\text{s}]$",
    ylabel="$\\text{Residual}$",
    title=f"Residual Components - {ref_sim.name}",
    labels=["Stationarity", "Equality", "Inequality", "Complementarity"]
)

# SQP Iterations
fig_sqp = plotter.generic_plot(
    t,
    *[s.sqp_iter for s in sims],
    xlabel="$t \\ [\\text{s}]$",
    ylabel="$\\text{SQP Iterations}$",
    title="SQP Iterations per MPC Solve",
    labels=[s.name for s in sims]
)

# Cost history
fig_cost = plotter.generic_plot(
    t,
    *[s.cost_history for s in sims],
    ylog=True,
    xlabel="$t \\ [\\text{s}]$",
    ylabel="$\\text{Cost}$",
    title="Cost History",
    labels=[s.name for s in sims]
)

# Timing Performance
fig_mpc_time = plotter.generic_plot(
    t, 
    *[s.mpc_time for s in sims],
    upper_bound=ref_sim.dt,
    xlabel="$t \\ [\\text{s}]$", 
    ylabel="$\\text{MPC Time} \\ [\\text{s}]$", 
    title=f"MPC Computation Time (dt = {ref_sim.dt}s)", 
    labels=[s.name for s in sims]
)

fig_integrator_time = plotter.generic_plot(
    t, 
    *[s.integration_time for s in sims],
    xlabel="$t \\ [\\text{s}]$", 
    ylabel="$\\text{Integration Time} \\ [\\text{s}]$", 
    title="Integration Time", 
    labels=[s.name for s in sims]
)

# Stacked bar chart for average time usage
avg_time_data = {
    "Avg. MPC Time": [s.get_summary()['avg_mpc_time'] for s in sims],
    "Avg. Integration Time": [s.get_summary()['avg_integration_time'] for s in sims],
}
fig_avg_time_usage = plotter.stacked_bar_plot(
    data=avg_time_data,
    group_labels=[s.name for s in sims],
    xlabel="Simulation",
    ylabel="Average Time [s]",
    title="Average Time Usage per Step"
)
'''
fig_grid_search = plotter.grid_search_heatmap(
    sims=sims,
    param_x='prediction_horizon',
    param_y='dt',
    metric='rmse_e1',
    xlabel='Prediction Horizon',
    ylabel='Solver Type',
    title='$e_1$ RMSE vs. MPC Parameters'
) '''


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("=== Solver Statistics Summary ===")
for s in sims:
    stats = s.get_summary()
    print(f"{s.name:12} | total_sqp={stats['total_sqp_iterations']:6} | "
          f"avg_sqp={stats['avg_sqp_iterations']:5.2f} | " +
          f"total_sim_time={stats['total_computation_time']:7.3f}s | "
          f"failures={stats['num_failures']:3}")
    print(f"           | RMSE: e1={stats['rmse_e1']:.4f}, e2={stats['rmse_e2']:.4f}, e3={stats['rmse_e3']:.4f}, e4={stats['rmse_e4']:.4f}")
    print(f"           | ITSE: e1={stats['itse_e1']:.4f}, e2={stats['itse_e2']:.4f}, e3={stats['itse_e3']:.4f}, e4={stats['itse_e4']:.4f}")

# ============================================================================
# GENERATE HTML REPORT
# ============================================================================
task_figs = [fig_rmse, fig_itse, fig_total_time, fig_e1, fig_e2, fig_e3, fig_e4, 
             fig_joints_q, fig_joints_qdot, fig_joints_qdotdot]
solver_figs = [fig_kkt, fig_res_components, fig_sqp, 
               fig_integrator_time, fig_mpc_time, fig_avg_time_usage, fig_cost]

plotter.gen_html_report(
    task_figs=task_figs,
    solver_figs=solver_figs,
    video_folder="video",
    title=f"6DOF Robot Manipulator - Comparative Study ({len(sims)} simulations)",
    filename="report.html"
)
report_path = plotter.gen_html_report(
    task_figs=task_figs,
    solver_figs=solver_figs,
    title="My run",
    filename="run_01.html",
    open_browser=True,
)
print("Report:", report_path)

'''
print(f"\n{'='*60}")
print(f"Report generated: report.html")
print(f"{'='*60}")'''