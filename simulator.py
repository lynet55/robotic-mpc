from loader import UrdfLoader as urdf
from visualizer import MeshCatVisualizer as robot_visualizer
from prediction_model import SixDofRobot as prediction_robot_6dof
from simulation_model import Robot as simulation_robot_6dof

from trajectory_optimizer import MPC as model_predictive_control
from surface import Surface
import numpy as np
from itertools import product
import time
from functools import cached_property
import sys


class Simulator:
    """Simplified simulator with clean, cached API for data access."""
    
    def __init__(self, 
                 # Core simulation parameters
                 robot_name,
                 dt, simulation_time, prediction_horizon,
                 q_0, qdot_0, wcv, q_min, q_max, 
                 qdot_min, qdot_max, 
                 
                 # Surface geometry
                 surface_limits, surface_origin, surface_orientation_rpy,
                 w_qddot,
                 
                 # Optional parameters with clear defaults
                 surface_coeffs=None,
                 solver_options=None,
                 w_u=0.01,
                 px_ref=0.40,
                 vy_ref=0.20,
                 scene=True):
        
        # Store all parameters
        self.name = robot_name
        self.dt = dt
        self.simulation_time = simulation_time
        self.Nsim = int(simulation_time/dt)
        self.prediction_horizon = prediction_horizon
        
        self.q_0 = q_0
        self.qdot_0 = qdot_0
        self.wcv = wcv
        self.q_min = q_min
        self.q_max = q_max
        self.qdot_min = qdot_min
        self.qdot_max = qdot_max
        self.w_u=w_u
        self.w_qddot=w_qddot
        self.initial_state = np.hstack((self.q_0, self.qdot_0))
        
        self.px_ref = px_ref
        self.vy_ref = vy_ref
        self.verbose = verbose
        
        # Initialize tracking arrays
        self.mpc_time = np.zeros(self.Nsim)
        self.integration_time = np.zeros(self.Nsim)
        self.sqp_iter = np.zeros(self.Nsim, dtype=int)
        self.solver_status = np.zeros(self.Nsim, dtype=int)
        self.residuals = np.zeros((self.Nsim, 4))
        self.solver_time = np.zeros(self.Nsim)
        self.cost_history = np.zeros(self.Nsim)
        
        # Create surface with coefficients
        self.surface = Surface(
            position=surface_origin,
            orientation_rpy=surface_orientation_rpy,
            limits=surface_limits,
            coefficients=surface_coeffs
        )
        
        # Initialize robot and models
        self.robot_loader = urdf(self.name)
        
        self.simulation_model = simulation_robot_6dof(
            urdf_loader=self.robot_loader,
            z0=self.initial_state,
            u0=self.qdot_0,
            dt=self.dt,
            Nsim=self.Nsim,
            wcv=self.wcv,
            integration_method="RK4"
        )
        
        self.prediction_model = prediction_robot_6dof(
            urdf_loader=self.robot_loader,
            Ts=self.dt,
            Wcv=self.wcv
        )
        self.translation = self.prediction_model.translation
        
        # Create MPC
        self.mpc = model_predictive_control(
            surface=self.surface,
            initial_state=self.initial_state,
            model=self.prediction_model,
            N_horizon=self.prediction_horizon,
            Tf=self.dt*self.prediction_horizon,
            qmin=self.q_min,
            qmax=self.q_max,
            dq_min=self.qdot_min,
            dq_max = self.qdot_max,
            w_u=self.w_u,
            w_qddot=self.w_qddot,
            px_ref=self.px_ref,
            vy_ref=self.vy_ref
        )
        
        # Apply solver options if provided
        if solver_options:
            self._apply_solver_options(solver_options)

        # Setup solver
        self.mpc.finalize_solver()
        
        # Setup visualization
        if scene:
            self.scene = robot_visualizer(self.robot_loader)
            self._setup_visualization()
        else:
            self.scene = None
        
        # Cache invalidation flag
        self._data_computed = False

    def _apply_solver_options(self, options):
        """Apply solver-specific options to the MPC."""
        for key, value in options.items():
            if hasattr(self.mpc.ocp.solver_options, key):
                setattr(self.mpc.ocp.solver_options, key, value)
            else:
                print(f"Warning: Unknown solver option '{key}'")

    def _setup_visualization(self):
        """Setup visualization elements."""
        self.scene.add_surface_from_casadi(
            self.surface.get_surface_function(),
            x_limits=self.surface.limits[0],
            y_limits=self.surface.limits[1],
            resolution=80,
            path="surfaces/quadratic_surface",
            color=0x3399FF,
            opacity=0.4,
            origin=self.surface.position,
            orientation_rpy=self.surface.orientation_rpy,
        )
        self.scene.add_dashed_isoline_on_surface(
            casadi_surface_function=self.surface.get_surface_function(),
            x_const=self.px_ref,
            y_limits=self.surface.limits[1],
            n_samples=500,
            dash_every=6,
            path="lines/px_ref_dashed",
            color=0x000000,        # arancio scuro (non confligge con triadi RGB)
            line_width=5.0,
            origin=self.surface.position,
            orientation_rpy=self.surface.orientation_rpy,
        )

        ee_position_0 = self.simulation_model.ee_position(0)
        self.scene.add_triad(
            position=ee_position_0,
            orientation_rpy=self.simulation_model.ee_orientation_euler(0),
            path="frames/end_effector_frame",
            scale=0.2,
            line_width=2.0
        )
        
        self.traj_points = [ee_position_0.copy()]
        self.traj_array = np.vstack(self.traj_points)
        
        self.scene.add_line(
            self.traj_array,
            path="lines/ee_trajectory",
            color=0xFF0000,
            line_width=2.0
        )
        z_ref = self.surface.get_point_on_surface(self.px_ref, 0)
        self.scene.add_point(
        position=np.array([self.px_ref, 0, z_ref]),
        path="points/Px_ref",
        color=0x000000,   # arancio scuro (non confligge con RGB)
        radius=0.010,
        opacity=0.8
        )


    def _invalidate_cache(self):
        """Invalidate computed property caches."""
        self._data_computed = False
        # Clear cached_property caches
        for attr in ['errors', 'metrics', 'solver_stats', 'timings']:
            if attr in self.__dict__:
                delattr(self, attr)

    def run(self):
        """Run the simulation."""
        self._invalidate_cache()
        
        for i in range(self.Nsim):
            start_time = time.time()
            # State Feedback
            current_state = self.simulation_model.state(i) 

            # MPC solve
            mpc_start_time = time.time()
            self.mpc.solver.set(0, 'lbx', current_state)
            self.mpc.solver.set(0, 'ubx', current_state)
            status = self.mpc.solver.solve() # solver call
            u = self.mpc.solver.get(0, "u")
            self.mpc_time[i] = time.time() - mpc_start_time

            # Store solver stats
            self.solver_status[i] = status # status = 0 is the only one with convergence 
            self.sqp_iter[i] = self.mpc.solver.get_stats('sqp_iter') # number of SQP iterations
            self.residuals[i, :] = self.mpc.solver.get_residuals() # returns [res_stat, res_eq, res_ineq, res_comp].  
            self.solver_time[i] = self.mpc.solver.get_stats('time_tot') # total CPU time previous call
            self.cost_history[i] = self.mpc.solver.get_cost() # objective value at the current solution
            
            # Integration
            integration_start_time = time.time()
            self.simulation_model.update(current_state, u, i)
            self.integration_time[i] = time.time() - integration_start_time
            
            #self._update_visualization(i)
            #time.sleep(1000)

            # Visualization update
            if self.scene and i % 10 == 0:
                self._update_visualization(i)
            
            # Minimal progress print every 100 steps
            if self.verbose and i % 100 == 0:
                pct = (i + 1) / self.Nsim * 100
                print(f"\r  t={self.dt*i:.2f}s [{pct:5.1f}%]", end="", flush=True)
            
            # Timing control
            time_spent = time.time() - start_time
            if time_spent < self.dt:
                time.sleep(self.dt - time_spent)
        
        if self.verbose:
            print()  # Newline after progress
        self._data_computed = True

    def _update_visualization(self, step):
        """Update visualization at given step."""
        q1, q2, q3, q4, q5, q6 = self.simulation_model.joint_angles(step+1)
        ee_pos = self.simulation_model.ee_position(step+1)
        
        self.scene.set_joint_angles({
            "shoulder_pan_joint": q1,
            "shoulder_lift_joint": q2,
            "elbow_joint": q3,
            "wrist_1_joint": q4,
            "wrist_2_joint": q5,
            "wrist_3_joint": q6
        })
        
        self.traj_points.append(ee_pos.copy())
        self.traj_array = np.vstack(self.traj_points)
        
        self.scene.update_triad("frames/end_effector_frame", 
                               position=ee_pos, 
                               orientation_rpy=self.simulation_model.ee_orientation_euler(step+1))
        self.scene.update_line(path="lines/ee_trajectory", points=self.traj_array)
    
    @cached_property
    def errors(self):
        if not self._data_computed:
            raise RuntimeError("Must call run() before accessing errors")

        # Logged EE pose in WORLD: [p(3); R_flat(9)] for each step
        p_ee_all = self.simulation_model._ee_pose_log[:3, :]      
        R_flat_all = self.simulation_model._ee_pose_log[3:12, :] 


        # |OUTPUT outside simulation loop|
        #pose, v_ee = self.simulation_model.simulate_output()
        #p_ee_all = pose[:3,:]
        #R_flat_all = pose[:3,:]

        n_steps = p_ee_all.shape[1]

        e1 = np.zeros(n_steps)
        e2 = np.zeros(n_steps)
        e3 = np.zeros(n_steps)
        e4 = np.zeros(n_steps)
        e5 = np.zeros(n_steps)
        p_task_z_all = np.zeros(n_steps)

        # EE - task fixed rotation
        R_ee_t = np.array([
            [1.0,  0.0,  0.0],
            [0.0, 1.0,  0.0],
            [0.0,  0.0, 1.0],
        ])

        t_ee = np.asarray(self.translation).reshape(3,)  

        n_fun = self.surface.get_normal_vector_casadi()
        S_fun = self.surface.get_surface_function()

        for i in range(n_steps):
            p_ee = p_ee_all[:, i]                 
            R_w_ee = R_flat_all[:, i].reshape(3,3)  
            vee = self.simulation_model.ee_velocity(i)[:]
            v_ee_w=vee[:3]
            w_ee_w=vee[3:6]

            # Translation vector in WORLD
            translation_w = R_w_ee @ t_ee

            # Task frame in WORLD
            R_w_t = R_w_ee @ R_ee_t
            # Origin of task frame in WORLD
            p_t = p_ee + R_w_ee @ t_ee

            # Task frame speed in TASK
            v_task = R_w_t @ (v_ee_w + w_ee_w.T @ translation_w)

            # task axes expressed in WORLD (columns!)
            R_task_y = R_w_t[:, 1]
            R_task_z = R_w_t[:, 2]

            # task position
            p_task_x, p_task_y, p_task_z = p_t

            # surface normal in WORLD at (x,y) task coords
            n = np.array(n_fun(p_task_x, p_task_y)).reshape(3,)

            # task constraints
            z_surf = float(S_fun(p_task_x, p_task_y))
            g1 = z_surf - p_task_z              # S(x,y) - z
            g2 = float(n @ R_task_z)            # alignment
            g3 = float(R_task_y[0])             # x-component of task y-axis
            g4 = p_task_x
            g5 = v_task[1]

            e1[i] = g1
            e2[i] = 1.0 - g2
            e3[i] = g3
            e4[i] = self.px_ref - g4
            e5[i] = self.vy_ref - g5
            p_task_z_all[i] = p_task_z # --> to plot task_z VS surf_z along p_ee_x

        return {'e1': e1, 'e2': e2, 'e3': e3, 'e4': e4, 'e5': e5, 'p_task_z': p_task_z_all, 'p_ee_y': p_ee_all[1,:]}

    
    @cached_property
    def metrics(self):
        """
        Compute all performance metrics in one pass (cached).
        
        Returns:
            dict with keys:
            - weighted_rmse: metric for all errors
            - itse: dict with e1, e2, e3, e4, e5
        """
        if not self._data_computed:
            raise RuntimeError("Must call run() before accessing metrics")
        
        errors = self.errors

        # Calculate ITSE for all errors
        itse = {}
        for key in ['e1', 'e2', 'e3', 'e4', 'e5']:
            error_signal = errors[key]
            num_steps = len(error_signal)
            time_vector = np.arange(num_steps) * self.dt
            integrand = time_vector * (error_signal**2)
            itse[key] = np.sum(integrand) * self.dt

        e1, e2, e3, e4, e5 = errors['e1'], errors['e2'], errors['e3'], errors['e4'], errors['e5']
        
        # Calculate RMSE for each error (scalars)
        rmse = {
            'e1': float(np.sqrt(np.mean(e1**2))),
            'e2': float(np.sqrt(np.mean(e2**2))),
            'e3': float(np.sqrt(np.mean(e3**2))),
            'e4': float(np.sqrt(np.mean(e4**2))),
            'e5': float(np.sqrt(np.mean(e5**2))),
        }
        
        # Weighted RMSE using MPC cost weights
        w1, w2, w3, w4, w5 = self.mpc.w_origin_task, self.mpc.w_normal_alignment_task, self.mpc.w_x_alignment_task, self.mpc.w_fixed_x_task, self.mpc.w_fixed_vy_task
        weighted_rmse = float(np.sqrt(np.mean(w1*e1**2 + w2*e2**2 + w3*e3**2 + w4*e4**2 + w5*e5**2)))
        
        return {
            'weighted_rmse': weighted_rmse,
            'rmse': rmse,
            'itse': itse
        }
    
    @cached_property
    def solver_stats(self):
        """
        Get solver statistics (cached).
        
        Returns:
            dict with solver performance metrics
        """
        if not self._data_computed:
            raise RuntimeError("Must call run() before accessing solver_stats")
        
        kkt_residuals = np.max(self.residuals, axis=1)
        
        return {
            'sqp_iterations': self.sqp_iter,
            'solver_status': self.solver_status,
            'residuals': self.residuals,
            'kkt_residuals': kkt_residuals,
            'res_stat': self.residuals[:, 0],
            'res_eq': self.residuals[:, 1],
            'res_ineq': self.residuals[:, 2],
            'res_comp': self.residuals[:, 3],
            'cost_history': self.cost_history,
            'total_sqp_iterations': int(np.sum(self.sqp_iter)),      # Negligable
            'avg_sqp_iterations': float(np.mean(self.sqp_iter)),     # Negligable
            'num_failures': int(np.sum(self.solver_status != 0)),    # Negligable
            'total_solver_time': float(np.sum(self.solver_time)),# Negligable
            'max_kkt_residual': float(np.max(kkt_residuals))         # Negligable (time istant could be added)
        }
    
    @cached_property
    def timings(self):
        """
        Get timing information (cached).
        
        Returns:
            dict with timing data
        """
        if not self._data_computed:
            raise RuntimeError("Must call run() before accessing timings")
        
        total_time = self.mpc_time + self.integration_time
        
        return {
            # Along simulation [arrays]
            'mpc_time': self.mpc_time,
            'integration_time': self.integration_time,
            'solver_time': self.solver_time, # VS mpc_time (should be the almost the same, with solver_time lower)
            'total_computation_time': total_time,    
            # Mean along the simulation [scalar]
            'avg_mpc_time': float(np.mean(self.mpc_time)),
            'avg_solver_time': float(np.mean(self.solver_time)),
            'avg_integration_time': float(np.mean(self.integration_time)),
            'avg_total_time': float(np.mean(total_time)),
            # Sum along the simulation [scalar]
            'computational_time_sim': float(np.sum(total_time)) # VS simulation time
        }

    def get_data(self):
        """
        Get raw simulation time-series data.
        
        Returns:
            dict with time, states, controls, and derived quantities
        """
        if not self._data_computed:
            raise RuntimeError("Must call run() before accessing data")
        
        q = self.simulation_model.z[:6, :]       
        qdot = self.simulation_model.z[6:, :]     
        u = self.simulation_model.u             

        
        wcv = np.asarray(self.wcv).reshape(-1, 1)  
        qddot = -wcv * qdot + wcv * u              

        Ts = self.dt  
        qddot_fd = np.empty_like(qdot)           
        
        qddot_fd[:, :-1] = (qdot[:, 1:] - qdot[:, :-1]) / Ts

        
        qddot_fd[:, -1] = qddot_fd[:, -2]

        #
        return {
            'time': np.arange(q.shape[1]) * self.dt,
            'q': q,
            'qdot': qdot,
            'qddot': qddot,
            'qddot_fd': qddot_fd,   # <--- added
            'u': u,
            'ee_pose': self.simulation_model._ee_pose_log,
            'px_ref': self.mpc.px_ref,
            'vy_ref': self.mpc.vy_ref,
            'N': self.prediction_horizon
        }


    
    def get_analysis(self):
        """
        Get all derived analysis data (errors, metrics, solver stats, timings).
        
        Returns:
            dict combining errors, metrics, solver_stats, and timings
        """
        if not self._data_computed:
            raise RuntimeError("Must call run() before accessing analysis")
        
        return {
            **self.errors,
            **self.metrics,
            **self.solver_stats,
            **self.timings
        }
    
    def get_summary(self):
        """
        Get compact summary statistics for comparison across simulations.
        
        Returns:
            dict with key performance indicators
        """
        if not self._data_computed:
            raise RuntimeError("Must call run() before accessing summary")
        
        m = self.metrics
        s = self.solver_stats
        t = self.timings
        
        return {
            # Error metrics
            'rmse_e1': m['rmse']['e1'],
            'rmse_e2': m['rmse']['e2'],
            'rmse_e3': m['rmse']['e3'],
            'rmse_e4': m['rmse']['e4'],
            'rmse_e5': m['rmse']['e5'],
            'itse_e1': m['itse']['e1'],
            'itse_e2': m['itse']['e2'],
            'itse_e3': m['itse']['e3'],
            'itse_e4': m['itse']['e4'],
            'itse_e5': m['itse']['e5'],
            'weighted_rmse': m['weighted_rmse'],
            # Solver performance
            'total_sqp_iterations': s['total_sqp_iterations'],
            'avg_sqp_iterations': s['avg_sqp_iterations'],
            'num_failures': s['num_failures'],
            'max_kkt_residual': s['max_kkt_residual'],
            'total_solver_time': s['total_solver_time'],
            # Timing
            'avg_mpc_time': t['avg_mpc_time'],
            'avg_solver_time': t['avg_solver_time'],
            'avg_integration_time': t['avg_integration_time'],
            'total_computation_time': t['computational_time_sim']
        }

class SimulationManager:
    """Manages parameter sweeps and grid searches with simplified configuration."""
    
    def __init__(self, base_config):
        """
        Initialize with base configuration.
        
        Args:
            base_config: Dict with all required Simulator parameters
        """
        self.base_config = base_config.copy()
        self.simulations = []
        
    def sweep(self, param_path, values, name_template=None):
        """
        Sweep a single parameter, supporting nested paths for surface coefficients.
        
        Args:
            param_path: Parameter name or path like 'surface_coeffs.a'
            values: List of values to sweep
            name_template: Optional naming template (e.g., 'coeff_a={}')
        
        Example:
            manager.sweep('surface_coeffs.a', [-0.2, -0.1, 0.0])
            manager.sweep('prediction_horizon', [50, 100, 200])
        """
        for val in values:
            config = self.base_config.copy()
            
            # Handle nested parameters (e.g., surface_coeffs.a)
            if '.' in param_path:
                parts = param_path.split('.')
                if parts[0] == 'surface_coeffs':
                    if 'surface_coeffs' not in config:
                        config['surface_coeffs'] = {}
                    else:
                        config['surface_coeffs'] = config['surface_coeffs'].copy()
                    config['surface_coeffs'][parts[1]] = val
            else:
                config[param_path] = val
            
            name = name_template.format(val) if name_template else f"{param_path}={val}"
            self.simulations.append({'name': name, 'config': config})
    
    def grid_search(self, param_grid, surface_coeff_sets=None, name_template=None):
        """
        Grid search over multiple parameters with optional surface coefficient sets.
        
        Args:
            param_grid: Dict mapping param paths to value lists
            surface_coeff_sets: List of dicts with complete surface coefficient sets
            name_template: Optional function to generate names
        """
        param_paths = list(param_grid.keys())
        value_lists = [param_grid[path] for path in param_paths]
        
        if surface_coeff_sets is None:
            surface_coeff_sets = [None]
        
        for coeff_idx, coeff_set in enumerate(surface_coeff_sets):
            for values in product(*value_lists):
                config = self.base_config.copy()
                
                # Apply all parameter values
                for param_path, val in zip(param_paths, values):
                    if '.' in param_path:
                        parts = param_path.split('.')
                        if parts[0] not in config:
                            config[parts[0]] = {}
                        config[parts[0]][parts[1]] = val
                    else:
                        config[param_path] = val
                
                # Apply surface coefficient set if provided
                if coeff_set is not None:
                    config['surface_coeffs'] = coeff_set.copy()
                
                # Generate name
                if name_template:
                    try:
                        name = name_template(dict(zip(param_paths, values)), coeff_idx)
                    except TypeError:
                        name = name_template(dict(zip(param_paths, values)))
                else:
                    param_str = '_'.join([f"{p.split('.')[-1]}={v}" for p, v in zip(param_paths, values)])
                    if coeff_set is not None:
                        name = f"coeffs{coeff_idx}_{param_str}"
                    else:
                        name = param_str
                
                self.simulations.append({'name': name, 'config': config})
    
    def run_all(self, return_results=True, verbose=True):
        """
        Run all queued simulations.
        
        Args:
            return_results: If True, return list of result dicts
            verbose: If True, print progress status
        """
        results = []
        completed = []
        sim_times = []
        total_sims = len(self.simulations)
        
        for i, sim_spec in enumerate(self.simulations):
            # Print persistent header
            if verbose:
                self._print_status(i, total_sims, sim_spec['name'], completed, sim_times)
            
            sim_start = time.time()
            # Override verbose in config to match manager setting
            config = sim_spec['config'].copy()
            config['verbose'] = verbose
            sim = Simulator(**config)
            sim.name = sim_spec['name']
            sim.run()
            sim_elapsed = time.time() - sim_start
            
            sim_times.append(sim_elapsed)
            completed.append((sim_spec['name'], sim_elapsed))
            
            if return_results:
                results.append({
                    'name': sim.name,
                    'simulator': sim,
                    'data': sim.get_data(),
                    'analysis': sim.get_analysis(),
                    'summary': sim.get_summary()
                })
        
        # Final status
        if verbose:
            self._print_status(total_sims, total_sims, None, completed, sim_times, done=True)
        return results if return_results else None
    
    def _print_status(self, current_idx, total, current_name, completed, sim_times, done=False):
        """Print persistent status overview."""
        # Calculate ETA
        if sim_times:
            avg_time = sum(sim_times) / len(sim_times)
            remaining = total - current_idx
            eta = avg_time * remaining
            eta_str = f"ETA: {eta:.1f}s" if eta < 120 else f"ETA: {eta/60:.1f}m"
        else:
            eta_str = "ETA: --"
        
        # Clear and print header
        print(f"\033[2J\033[H", end="")  # Clear screen, cursor to top
        
        # Status line
        if done:
            print(f"═══ DONE: {total} simulations ═══")
        else:
            bar_len = 20
            filled = int(bar_len * current_idx / total)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"═══ [{current_idx+1}/{total}] {bar} {eta_str} ═══")
            print(f"▶ {current_name}")
        
        # Completed list (last 8 max to keep it compact)
        if completed:
            print("─" * 40)
            show = completed[-8:] if len(completed) > 8 else completed
            if len(completed) > 8:
                print(f"  ... +{len(completed) - 8} earlier")
            for name, elapsed in show:
                print(f"  ✓ {name} ({elapsed:.1f}s)")
        
        print("─" * 40)
        sys.stdout.flush()
    
    def clear(self):
        """Clear all queued simulations."""
        self.simulations = []


if __name__ == "__main__":
    
    # Base configuration
    base_config = {
        'robot_name': 'ur10',
        'dt': 0.0004,
        'simulation_time': 6.0,
        'prediction_horizon': 200,
        'surface_limits': ((-2, 2), (-2, 2)),
        'surface_origin': np.array([0.0, 0.0, 0.0]),
        'surface_orientation_rpy': np.array([0.0, 0.0, 0.0]),
        'q_0': np.array([np.pi/3, -np.pi/3, np.pi/4, -np.pi/2, -np.pi/2, 0.0]),
        'qdot_0': np.array([2, 0, 0, -1, 1, 1]),
        'wcv': np.array([228.9, 262.09, 517.3, 747.44, 429.9, 1547.76], dtype=float),
        'q_min': np.array([-2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi], dtype=float),
        'q_max': np.array([+2*np.pi, +2*np.pi, +2*np.pi, +2*np.pi, +2*np.pi, +2*np.pi], dtype=float),
        'qdot_min': np.array([-2.16, -2.16, -np.pi, -3.20, -3.20, -3.20], dtype=float),
        'qdot_max': np.array([2.16, 2.16, np.pi, 3.20, 3.20, 3.20], dtype=float),
        'scene': True
        }
    
    # Example: Single simulation
    sim = Simulator(**base_config)
    sim.run()
    
    # Quick summary
    summary = sim.get_summary()
    print(f"\nSummary: Max RMSE={summary['max_combined_rmse']:.6f}, \n"
          f"Failures={summary['num_failures']}, \n"
          f"Avg MPC time={summary['avg_mpc_time']*1000:.2f}ms, \n"
          f"Avg plant integration time={summary['avg_integration_time']*1000:.2f}ms, \n"
          f"Avg solver time={summary['avg_solver_time']*1000:.2f}ms, \n")
    

    
    # Get raw data for plotting
    data = sim.get_data()
    print(f"\nData shapes: q={data['q'].shape}, u={data['u'].shape}")
    
    # Full analysis if needed (computed once, cached)
    analysis = sim.get_analysis()
    print(f"Analysis keys: {list(analysis.keys())[:5]}...")  # Show first 5 keys
    
    '''
    # Batch simulations
    manager = SimulationManager(base_config)
    manager.sweep('prediction_horizon', [50, 100, 200], name_template='H={}')
    results = manager.run_all()
    
    # Compare
    print(f"\n{'Name':<12} {'RMSE':<10} {'SQP':<8} {'Fail':<6}")
    print("─" * 36)
    for r in results:
        s = r['summary']
        print(f"{r['name']:<15} {s['max_combined_rmse']:<12.6f} "
              f"{s['total_sqp_iterations']:<12} {s['num_failures']:<10}")'''