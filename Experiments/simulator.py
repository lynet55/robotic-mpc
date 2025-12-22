from Infrastructure.loader import UrdfLoader as urdf
from Infrastructure.visualizer import MeshCatVisualizer as robot_visualizer

from models.prediction_model import SixDofRobot as prediction_robot_6dof
from models.simulation_model import Robot as simulation_robot_6dof

from mpc.trajectory_optimizer import MPC as model_predictive_control
from mpc.surface import Surface

from Reporting.plotter import Plotter

import numpy as np
from itertools import product
import time
from functools import cached_property


class Simulator:
    """Simplified simulator with clean, cached API for data access."""
    
    def __init__(self, 
                 # Core simulation parameters
                 dt, simulation_time, prediction_horizon,
                 q_0, qdot_0, wcv, q_min, q_max, 
                 qdot_min, qdot_max,
                 
                 # Surface geometry
                 surface_limits, surface_origin, surface_orientation_rpy,
                 
                 # Optional parameters with clear defaults
                 surface_coeffs=None,
                 solver_options=None,
                 px_ref=0.40,
                 vy_ref=-0.20,
                 scene=True):
        
        # Store all parameters
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
        self.initial_state = np.hstack((self.q_0, self.qdot_0))
        
        self.px_ref = px_ref
        self.vy_ref = vy_ref
        
        # Initialize tracking arrays
        self.mpc_time = np.zeros(self.Nsim)
        self.integration_time = np.zeros(self.Nsim)
        self.sqp_iter = np.zeros(self.Nsim, dtype=int)
        self.solver_status = np.zeros(self.Nsim, dtype=int)
        self.residuals = np.zeros((self.Nsim, 4))
        self.solver_time_tot = np.zeros(self.Nsim)
        self.cost_history = np.zeros(self.Nsim)
        
        # Create surface with coefficients
        self.surface = Surface(
            position=surface_origin,
            orientation_rpy=surface_orientation_rpy,
            limits=surface_limits,
            coefficients=surface_coeffs
        )
        
        # Initialize robot and models
        self.robot_loader = urdf('ur5')
        
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
        self.translation = self.prediction_model.translation_array
        
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
            px_ref=self.px_ref,
            vy_ref=self.vy_ref
        )
        
        # Apply solver options if provided
        if solver_options:
            self._apply_solver_options(solver_options)
        
        print("qp_solver =", self.mpc.ocp.solver_options.qp_solver)
        print("nlp_solver_type =", self.mpc.ocp.solver_options.nlp_solver_type)


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
            opacity=0.6,
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
            line_width=3.0,
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
            status = self.mpc.solver.solve() 
            u = self.mpc.solver.get(0, "u")
            self.mpc_time[i] = time.time() - mpc_start_time
            
            # Store solver stats
            self.solver_status[i] = status
            self.sqp_iter[i] = self.mpc.solver.get_stats('sqp_iter') # number of SQP iterations
            self.residuals[i, :] = self.mpc.solver.get_residuals() # returns [res_stat, res_eq, res_ineq, res_comp].  
            self.solver_time_tot[i] = self.mpc.solver.get_stats('time_tot') # total CPU time previous call
            self.cost_history[i] = self.mpc.solver.get_cost() # cost value of the current solution
            
            # Integration
            integration_start_time = time.time()
            self.simulation_model.update(current_state, u, i)
            self.integration_time[i] = time.time() - integration_start_time

            # Visualization update
            if self.scene and i % 10 == 0:
                self._update_visualization(i)
                print(f"Running time: {self.dt*i:.3f} s of {self.simulation_time} s")
            
            # Timing control
            time_spent = time.time() - start_time
            if time_spent < self.dt:
                time.sleep(self.dt - time_spent)
        
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
        p_ee_all = self.simulation_model._ee_pose_log[:3, :]      # (3, N+1)
        R_flat_all = self.simulation_model._ee_pose_log[3:12, :]  # (9, N+1)

        n_steps = p_ee_all.shape[1]

        e1 = np.zeros(n_steps)
        e2 = np.zeros(n_steps)
        e3 = np.zeros(n_steps)
        e4 = np.zeros(n_steps)
        e5 = np.zeros(n_steps)
        z_surface = np.zeros(n_steps)

        # EE -> task constant rotation
        R_ee_t = np.array([
            [1.0,  0.0,  0.0],
            [0.0, 1.0,  0.0],
            [0.0,  0.0, 1.0],
        ])

        t_ee = np.asarray(self.translation).reshape(3,)  # translation expressed in EE frame

        n_fun = self.surface.get_normal_vector_casadi()
        S_fun = self.surface.get_surface_function()

        for i in range(n_steps):
            p_ee = p_ee_all[:, i]                 # (3,)
            R_w_ee = R_flat_all[:, i].reshape(3,3)  # row-wise (3,3)

            # Task frame in WORLD
            R_w_t = R_w_ee @ R_ee_t
            p_t = p_ee + R_w_ee @ t_ee

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
            g5 = float(self.simulation_model.ee_velocity(i)[1])  # v_y in WORLD (or transform if you want v_t_y!)

            e1[i] = g1
            e2[i] = 1.0 - g2
            e3[i] = g3
            e4[i] = self.px_ref - g4
            e5[i] = self.vy_ref - g5
            z_surface[i] = z_surf

        return {'e1': e1, 'e2': e2, 'e3': e3, 'e4': e4, 'e5': e5, 'z_surface': z_surface}

    
    @cached_property
    def metrics(self):
        """
        Compute all performance metrics in one pass (cached).
        
        Returns:
            dict with keys:
            - rmse: dict with e1, e2, e3, e4, e5
            - itse: dict with e1, e2, e3, e4, e5
            - max_combined_rmse: maximum combined error across all constraints
        """
        if not self._data_computed:
            raise RuntimeError("Must call run() before accessing metrics")
        
        errors = self.errors
        
        # Calculate RMSE for all errors
        rmse = {}
        for key in ['e1', 'e2', 'e3', 'e4', 'e5']:
            error_signal = errors[key]
            rmse[key] = np.sqrt(np.sum(error_signal**2) * self.dt / self.simulation_time)

        
        # Calculate ITSE for all errors
        itse = {}
        for key in ['e1', 'e2', 'e3', 'e4', 'e5']:
            error_signal = errors[key]
            num_steps = len(error_signal)
            time_vector = np.arange(num_steps) * self.dt
            integrand = time_vector * (error_signal**2)
            itse[key] = np.sum(integrand) * self.dt
        
        # Combined RMSE (maximum across all constraints)
        e1, e2, e3, e4 = errors['e1'], errors['e2'], errors['e3'], errors['e4']
        max_combined_rmse = np.max(np.sqrt(e1**2 + e2**2 + e3**2 + e4**2))
        
        return {
            'rmse': rmse,
            'itse': itse,
            'max_combined_rmse': max_combined_rmse
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
            'sqp_iterations': self.sqp_iter,
            'total_sqp_iterations': int(np.sum(self.sqp_iter)),
            'avg_sqp_iterations': float(np.mean(self.sqp_iter)),
            'num_failures': int(np.sum(self.solver_status != 0)),
            'total_solver_time': float(np.sum(self.solver_time_tot)),
            'max_kkt_residual': float(np.max(kkt_residuals))
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
            'mpc_time': self.mpc_time,
            'integration_time': self.integration_time,
            'solver_time_tot': self.solver_time_tot,
            'total_computation_time': total_time,
            'avg_mpc_time': float(np.mean(self.mpc_time)),
            'avg_integration_time': float(np.mean(self.integration_time)),
            'total_time': float(np.sum(total_time))
        }
    
    # =========================================================================
    # HIGH-LEVEL API METHODS
    # =========================================================================
    
    def get_data(self):
        """
        Get raw simulation time-series data.
        
        Returns:
            dict with time, states, controls, and references
        """
        if not self._data_computed:
            raise RuntimeError("Must call run() before accessing data")
        
        return {
            'time': np.arange(0, self.simulation_model.z.shape[1]) * self.dt,
            'q': self.simulation_model.z[:6, :],
            'qdot': self.simulation_model.z[6:, :],
            'u': self.simulation_model.u,
            'ee_pose': self.simulation_model._ee_pose_log,
            'px_ref': self.mpc.px_ref,
            'vy_ref': self.mpc.vy_ref,
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
            'max_combined_rmse': m['max_combined_rmse'],
            # Solver performance
            'total_sqp_iterations': s['total_sqp_iterations'],
            'avg_sqp_iterations': s['avg_sqp_iterations'],
            'num_failures': s['num_failures'],
            'max_kkt_residual': s['max_kkt_residual'],
            'total_solver_time': s['total_solver_time'],
            # Timing
            'avg_mpc_time': t['avg_mpc_time'],
            'avg_integration_time': t['avg_integration_time'],
            'total_computation_time': t['total_time']
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
    
    def run_all(self, return_results=True):
        """
        Run all queued simulations.
        
        Args:
            return_results: If True, return list of result dicts
        """
        results = []
        
        for i, sim_spec in enumerate(self.simulations):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(self.simulations)}] Running: {sim_spec['name']}")
            print(f"{'='*60}")
            
            sim = Simulator(**sim_spec['config'])
            sim.name = sim_spec['name']
            sim.run()
            
            if return_results:
                results.append({
                    'name': sim.name,
                    'simulator': sim,
                    'data': sim.get_data(),
                    'analysis': sim.get_analysis(),
                    'summary': sim.get_summary()
                })
        
        print(f"\n{'='*60}\nCompleted {len(self.simulations)} simulations\n{'='*60}\n")
        return results if return_results else None
    
    def clear(self):
        """Clear all queued simulations."""
        self.simulations = []


# =============================================================================
# USAGE EXAMPLES - NEW CLEAN API
# =============================================================================

if __name__ == "__main__":
    
    # Base configuration
    base_config = {
        'dt': 0.001,
        'simulation_time': 2.0,
        'prediction_horizon': 200,
        'surface_limits': ((-2, 2), (-2, 2)),
        'surface_origin': np.array([0.0, 0.0, 0.0]),
        'surface_orientation_rpy': np.array([0.0, 0.0, 0.0]),
        'q_0': np.array([np.pi/3, -np.pi/3, np.pi/4, -np.pi/2, -np.pi/2, 0.0]),
        'qdot_0': np.array([2, 2, 2, 2, 2, 2]),
        'wcv': np.array([228.9, 262.09, 517.3, 747.44, 429.9, 1547.76], dtype=float),
        'q_min': np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi], dtype=float),
        'q_max': np.array([+2*np.pi, +2*np.pi, +np.pi, +2*np.pi, +2*np.pi, +2*np.pi], dtype=float),
        'qdot_min': np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi], dtype=float),
        'qdot_max': np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi], dtype=float),
        'scene': True
    }
    
    # Example: Single simulation with new API
    print("Running single simulation...")
    sim = Simulator(**base_config)
    sim.run()
    
    # Access data efficiently
    print("\n--- Accessing data with new API ---")
    
    # Get only what you need
    errors = sim.errors
    print(f"Surface tracking error (e1) range: [{errors['e1'].min():.4f}, {errors['e1'].max():.4f}]")
    
    metrics = sim.metrics
    print(f"RMSE values: e1={metrics['rmse']['e1']:.6f}, e2={metrics['rmse']['e2']:.6f}")
    
    solver_stats = sim.solver_stats
    print(f"Total SQP iterations: {solver_stats['total_sqp_iterations']}")
    
    # Get compact summary for comparison
    summary = sim.get_summary()
    print(f"\nSummary: Max RMSE={summary['max_combined_rmse']:.6f}, "
          f"Failures={summary['num_failures']}, "
          f"Avg MPC time={summary['avg_mpc_time']*1000:.2f}ms")
    
    # Get raw data for plotting
    data = sim.get_data()
    print(f"\nData shapes: q={data['q'].shape}, u={data['u'].shape}")
    
    # Full analysis if needed (computed once, cached)
    analysis = sim.get_analysis()
    print(f"Analysis keys: {list(analysis.keys())[:5]}...")  # Show first 5 keys
    
    # Batch simulations
    print("\n\n--- Running batch simulations ---")
    manager = SimulationManager(base_config)
    manager.sweep('prediction_horizon', [50, 100, 200], name_template='H={}')
    results = manager.run_all()
    
    # Compare results efficiently
    print("\n\nComparison across simulations:")
    print(f"{'Name':<15} {'Max RMSE':<12} {'Total SQP':<12} {'Failures':<10}")
    print("-" * 49)
    for r in results:
        s = r['summary']
        print(f"{r['name']:<15} {s['max_combined_rmse']:<12.6f} "
              f"{s['total_sqp_iterations']:<12} {s['num_failures']:<10}")