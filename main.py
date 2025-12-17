from simulator import Simulator as sim
import numpy as np

sim0 = sim(
    dt=0.001,
    prediction_horizon=200,
    simulation_time=3,
    surface_limits=((-2, 2), (-2, 2)),
    surface_origin=np.array([0.0, 0.0, 0.0]),
    surface_orientation_rpy=np.array([0.0, 0.0, 0.0]),
    qdot_0=np.array([2,2,2,2,2,2]),
    q_0=np.array([np.pi/3, -np.pi/3, np.pi/4, -np.pi/2, -np.pi/2, 0.0]),
    wcv=np.array([5,10,15,20,25,35]),
    scene=True
)

sim1 = sim(
    dt=0.001,
    prediction_horizon=10,
    simulation_time=3,
    surface_limits=((-2, 2), (-2, 2)),
    surface_origin=np.array([0.0, 0.0, 0.0]),
    surface_orientation_rpy=np.array([0.0, 0.0, 0.0]),
    qdot_0=np.array([2,2,2,2,2,2]),
    q_0=np.array([np.pi/3, -np.pi/3, np.pi/4, -np.pi/2, -np.pi/2, 0.0]),
    wcv=np.array([5,10,15,20,25,35]),
    scene=True
)

sim2 = sim(
    dt=0.001,
    prediction_horizon=50,
    simulation_time=3,
    surface_limits=((-2, 2), (-2, 2)),
    surface_origin=np.array([0.0, 0.0, 0.0]),
    surface_orientation_rpy=np.array([0.0, 0.0, 0.0]),
    qdot_0=np.array([2,2,2,2,2,2]),
    q_0=np.array([np.pi/3, -np.pi/3, np.pi/4, -np.pi/2, -np.pi/2, 0.0]),
    wcv=np.array([5,10,15,20,25,35]),
    scene=True
)
sim0.run()
sim1.run()
sim2.run()

def analyze_simulation(sim_obj):
    """
    Analyze simulation results and compute error signals.
    
    Args:
        sim_obj: Simulator object that has been run
        
    Returns:
        dict: Dictionary containing analysis results
    """
    results = sim_obj.get_results()
    
    t = results['time']
    q = results['q']
    u = results['u']
    
    # Transform to task space
    R_ee_t = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ])
    ee_xyz = results["ee_pose"][:3]
    ee_task = R_ee_t @ ee_xyz + np.array([0.0, 0.0, 1.0])[:, np.newaxis]
    
    px_ref = sim_obj.mpc.px_ref
    vy_ref = sim_obj.mpc.vy_ref
    
    # Compute error signals
    n_steps = ee_task.shape[1]
    e1 = np.zeros(n_steps)
    e2 = np.zeros(n_steps)
    e3 = np.zeros(n_steps)
    e4 = np.zeros(n_steps)
    z_surface = np.zeros(n_steps)
    rmse = np.zeros(n_steps)
    
    for i in range(n_steps):
        ee_x = ee_task[0, i]
        ee_y = ee_task[1, i]
        ee_z = ee_task[2, i]
        
        ee_vy = sim_obj.simulation_model.ee_velocity(i)[1]
        n = sim_obj.surface.get_normal_vector_casadi()(ee_x, ee_y)
        ee_rot_vector = sim_obj.simulation_model.ee_orientation(i)
        
        e1[i] = sim_obj.surface.get_point_on_surface(ee_x, ee_y) - ee_z  # e1 = s(x,y) - z
        e2[i] = px_ref - ee_x  # px_ref - px
        e3[i] = n.T @ ee_rot_vector  # orientation error
        e4[i] = vy_ref - ee_vy  # e4 = vy - v_ref
        z_surface[i] = sim_obj.surface.get_point_on_surface(ee_x, ee_y)
    
    max_rmse = np.max(np.sqrt(e1**2 + e2**2 + e3**2 + e4**2))

    # Acados solver statistics - now collected at EVERY MPC solve
    # residuals shape: (Nsim, 4) with columns [res_stat, res_eq, res_ineq, res_comp]
    residuals_all = results['residuals']
    sqp_iter_all = results['sqp_iter']  # SQP iterations per solve
    solver_status_all = results['solver_status']  # Solver status per solve
    solver_time_all = results['solver_time_tot']  # Acados solver time per solve
    mpc_time = results['mpc_time']  # Python-measured MPC time per solve
    integration_time = results['integration_time']
    cost_history = results['cost_history']
    
    # Compute KKT residual (max of all components) at each simulation step
    kkt_residuals = np.max(residuals_all, axis=1)
    
    # Individual residual components over time
    res_stat = residuals_all[:, 0]
    res_eq = residuals_all[:, 1]
    res_ineq = residuals_all[:, 2]
    res_comp = residuals_all[:, 3]
    
    # Summary statistics
    total_sqp_iter = np.sum(sqp_iter_all)
    avg_sqp_iter = np.mean(sqp_iter_all)
    total_solver_time = np.sum(solver_time_all)
    num_failures = np.sum(solver_status_all != 0)

    return {
        'time': t,
        'q': q,
        'qdot': results['qdot'],
        'u': u,
        'ee_task': ee_task,
        'px_ref': px_ref,
        'vy_ref': vy_ref,
        'e1': e1,
        'e2': e2,
        'e3': e3,
        'e4': e4,
        'z_surface': z_surface,
        # Solver stats (per simulation step)
        'sqp_iter': sqp_iter_all,
        'solver_status': solver_status_all,
        'solver_time': solver_time_all,
        'kkt_residuals': kkt_residuals,
        'cost_history': cost_history,
        'res_stat': res_stat,
        'res_eq': res_eq,
        'res_ineq': res_ineq,
        'res_comp': res_comp,
        # Timing
        'mpc_time': mpc_time,
        'integration_time': integration_time,
        'total_time': mpc_time + integration_time,
        # Summary stats
        'total_sqp_iter': total_sqp_iter,
        'avg_sqp_iter': avg_sqp_iter,
        'total_solver_time': total_solver_time,
        'num_failures': num_failures,
        'max_rmse': max_rmse
    }

    

# Analyze sim0
sim0_analysis = analyze_simulation(sim0)
sim1_analysis = analyze_simulation(sim1)
sim2_analysis = analyze_simulation(sim2)

# Plotting
from plotter import Plotter
plotter = Plotter()

t = sim0_analysis['time']
q = sim0_analysis['q']
qdot =  sim0_analysis['qdot']

fig_rmse = plotter.bar_plot(
    values=np.array([sim0_analysis['max_rmse'], sim1_analysis['max_rmse'], sim2_analysis['max_rmse']]),
    labels= ["H=sim0", "H=sim1", "H=sim2"],
    xlabel="Simulation",
    ylabel="Max RMSE",
    title=""
)

# Joint Angles Plot
fig_joints_q = plotter.joints(t, q, name="q", unit="rad")
# Joint Accelerations Plot
fig_joints_qdot = plotter.joints(t, qdot, name="\dot{q}", unit="rad/s")
# Joint Accelerations Plot
fig_joints_qdotdot = plotter.joints(t, qdot, name="\ddot{q}", unit="rad/s²")

# Control Input Plot
fig_u = plotter.generic_plot(
    t,
    sim0_analysis['u'][0],
    sim0_analysis['u'][1],
    sim0_analysis['u'][2],
    sim0_analysis['u'][3],
    sim0_analysis['u'][4],
    sim0_analysis['u'][5],
    xlabel="$t \\ [\\text{s}]$",
    ylabel="$u$ [rad/s]",
    title="$ sim_{0}$",
    labels=["$\dot{q}_{1}$", "$\dot{q}_{1}$","$\dot{q}_{3}$", "$\dot{q}_{4}$", "$\dot{q}_{5}$", "$\dot{q}_{6}$"])

# Constraint error plots
fig_e1 = plotter.generic_plot(
    t, 
    sim0_analysis['e1'], 
    sim1_analysis['e1'], 
    sim2_analysis['e1'], 
    xlabel="$t \\ [\\text{s}]$", 
    ylabel="$e_1 \\ [\\text{m}]$", 
    title="$e_1 = s(x,y) - z$", 
    labels=["$H=200$", "$H=10$", "$H=50$"]
)

fig_e2 = plotter.generic_plot(
    t, 
    sim0_analysis['e2'], 
    sim1_analysis['e2'], 
    sim2_analysis['e2'], 
    xlabel="$t \\ [\\text{s}]$", 
    ylabel="$e_2 \\ [\\text{m}]$", 
    title="$e_2 = p_{x,\\text{ref}} - p_x$", 
    labels=["$H=200$", "$H=10$", "$H=50$"]
)

fig_e3 = plotter.generic_plot(
    t, 
    sim0_analysis['e3'], 
    sim1_analysis['e3'], 
    sim2_analysis['e3'], 
    xlabel="$t \\ [\\text{s}]$", 
    ylabel="$e_3$", 
    title="$e_3 = n^T R_{\\text{ee}} v_{\\text{ee}}$", 
    labels=["$H=200$", "$H=10$", "$H=50$"]
)

fig_e4 = plotter.generic_plot(
    t, 
    sim0_analysis['e4'], 
    sim1_analysis['e4'], 
    sim2_analysis['e4'], 
    xlabel="$t \\ [\\text{s}]$", 
    ylabel="$e_4 \\ [\\text{m/s}]$", 
    title="$e_4 = v_{\\text{ref}} - v_{\\text{ee}}$", 
    labels=["$H=200$", "$H=10$", "$H=50$"]
)

# Control Input Plots
fig_u1 = plotter.generic_plot(
    t, 
    sim0_analysis['u'][0], 
    sim1_analysis['u'][0], 
    sim2_analysis['u'][0], 
    xlabel="$t \\ [\\text{s}]$", 
    ylabel="$u_1 \\ [\\text{rad/s}]$", 
    title="Control Input - Joint 1", 
    labels=["$H=200$", "$H=10$", "$H=50$"]
)

# KKT Residuals
fig_kkt = plotter.generic_plot(
    t, 
    sim0_analysis['kkt_residuals'], 
    sim1_analysis['kkt_residuals'], 
    sim2_analysis['kkt_residuals'], 
    ylog=True, 
    xlabel="$t \\ [\\text{s}]$", 
    ylabel="$\\text{KKT Residual (max)}$", 
    title="KKT Residuals over Simulation", 
    labels=["$H=200$", "$H=10$", "$H=50$"]
)

# Individual residual components
fig_res_components = plotter.generic_plot(
    t,
    sim0_analysis['res_stat'],
    sim0_analysis['res_eq'],
    sim0_analysis['res_ineq'],
    sim0_analysis['res_comp'],
    ylog=True,
    xlabel="$t \\ [\\text{s}]$",
    ylabel="$\\text{Residual}$",
    title="Residual Components ($H=200$)",
    labels=["Stationarity", "Equality", "Inequality", "Complementarity"]
)

# SQP Iterations
fig_sqp = plotter.generic_plot(
    t,
    sim0_analysis['sqp_iter'],
    sim1_analysis['sqp_iter'],
    sim2_analysis['sqp_iter'],
    ylog=True,
    xlabel="$t \\ [\\text{s}]$",
    ylabel="$\\text{SQP Iterations}$",
    title="SQP Iterations per MPC Solve",
    labels=["$H=200$", "$H=10$", "$H=50$"]
)

# Cost history
fig_cost = plotter.generic_plot(
    t,
    sim0_analysis['cost_history'],
    sim1_analysis['cost_history'],
    sim2_analysis['cost_history'],
    ylog=True,
    xlabel="$t \\ [\\text{s}]$",
    ylabel="$\\text{Cost}$",
    title="Cost History",
    labels=["$H=200$", "$H=10$", "$H=50$"]
)

# Timing Performance
fig_mpc_time = plotter.generic_plot(
    t, 
    sim0_analysis['mpc_time'], 
    sim1_analysis['mpc_time'], 
    sim2_analysis['mpc_time'], 
    xlabel="$t \\ [\\text{s}]$", 
    ylabel="$\\text{MPC Time} \\ [\\text{s}]$", 
    title="MPC Computation Time", 
    labels=["$H=200$", "$H=10$", "$H=50$"]
)

fig_integrator_time = plotter.generic_plot(
    t, 
    sim0_analysis['integration_time'], 
    sim1_analysis['integration_time'], 
    sim2_analysis['integration_time'], 
    xlabel="$t \\ [\\text{s}]$", 
    ylabel="$\\text{Integration Time} \\ [\\text{s}]$", 
    title="Integration Time", 
    labels=["$H=200$", "$H=10$", "$H=50$"]
)

# Summary statistics
print("=== Solver Statistics Summary ===")
print(f"Horizon 200: total_sqp={sim0_analysis['total_sqp_iter']}, avg_sqp={sim0_analysis['avg_sqp_iter']:.2f}, total_time={sim0_analysis['total_solver_time']:.4f}s, failures={sim0_analysis['num_failures']}")
print(f"Horizon 10:  total_sqp={sim1_analysis['total_sqp_iter']}, avg_sqp={sim1_analysis['avg_sqp_iter']:.2f}, total_time={sim1_analysis['total_solver_time']:.4f}s, failures={sim1_analysis['num_failures']}")
print(f"Horizon 50:  total_sqp={sim2_analysis['total_sqp_iter']}, avg_sqp={sim2_analysis['avg_sqp_iter']:.2f}, total_time={sim2_analysis['total_solver_time']:.4f}s, failures={sim2_analysis['num_failures']}")


# Generate HTML report
task_figs = [fig_rmse, fig_u, fig_e1, fig_e2, fig_e3, fig_e4, fig_joints_q, fig_joints_qdot, fig_joints_qdotdot]
solver_figs = [fig_kkt, fig_res_components, fig_sqp, fig_integrator_time, fig_mpc_time, fig_cost]

plotter.gen_html_report(
    task_figs=task_figs,
    solver_figs=solver_figs,
    video_folder="video",
    title="6DOF robot manipulator",
    filename="report.html"
)