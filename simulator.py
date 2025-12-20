from loader import UrdfLoader as urdf
from visualizer import MeshCatVisualizer as robot_visualizer
from prediction_model import SixDofRobot as prediction_robot_6dof
from simulation_model import Robot as simulation_robot_6dof
from trajectory_optimizer import MPC as model_predictive_control
from surface import Surface
from plotter import Plotter
import numpy as np
from itertools import product
import time
from functools import cached_property


class Simulator:
    """Simplified simulator with clean, cached API for data access."""
    
    def __init__(self, 
                 # Core simulation parameters
                 dt, simulation_time, prediction_horizon,
                 q_0, qdot_0, wcv,
                 
                 # Surface geometry
                 surface_limits, surface_origin, surface_orientation_rpy,
                 
                 # Optional parameters with clear defaults
                 surface_coeffs=None,
                 solver_options=None,
                 px_ref=0.40,
                 vy_ref=-0.40,
                 scene=True):
        
        # Store all parameters
        self.dt = dt
        self.simulation_time = simulation_time
        self.Nsim = int(simulation_time/dt)
        self.prediction_horizon = prediction_horizon
        
        self.q_0 = q_0
        self.qdot_0 = qdot_0
        self.wcv = wcv
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
            Wcv=self.wcv
        )
        
        # Create MPC
        self.mpc = model_predictive_control(
            surface=self.surface,
            initial_state=self.initial_state,
            model=self.prediction_model,
            N_horizon=self.prediction_horizon,
            Tf=self.dt*self.prediction_horizon,
            px_ref=self.px_ref,
            vy_ref=self.vy_ref
        )
        
        # Apply solver options if provided
        if solver_options:
            self._apply_solver_options(solver_options)
        
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
            orientation_rpy=self.simulation_model.ee_orientation(0),
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
            self.sqp_iter[i] = self.mpc.solver.get_stats('sqp_iter')
            self.residuals[i, :] = self.mpc.solver.get_residuals()
            self.solver_time_tot[i] = self.mpc.solver.get_stats('time_tot')
            self.cost_history[i] = self.mpc.solver.get_cost()
            
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
                               orientation_rpy=self.simulation_model.ee_orientation(step+1))
        self.scene.update_line(path="lines/ee_trajectory", points=self.traj_array)
    
    @cached_property
    def errors(self):
        """
        Compute all tracking errors in one pass (cached).
        
        Returns:
            dict with keys: e1, e2, e3, e4, e5, z_surface
            - e1: Surface tracking error (z_surface - z_ee)
            - e2: Orientation alignment error (1 - n^T @ rot_z)
            - e3: Orientation cross error (n^T @ rot_y)
            - e4: X-position tracking error (px_ref - px)
            - e5: Y-velocity tracking error (vy_ref - vy)
            - z_surface: Surface height at each step
        """
        if not self._data_computed:
            raise RuntimeError("Must call run() before accessing errors")
        
        # Transform to task space
        R_ee_t = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ])
        ee_xyz = self.simulation_model._ee_pose_log[:3]
        ee_task = R_ee_t @ ee_xyz + np.array([0.0, 0.0, 1.0])[:, np.newaxis]
        n_steps = ee_task.shape[1]
        
        e1 = np.zeros(n_steps)
        e2 = np.zeros(n_steps)
        e3 = np.zeros(n_steps)
        e4 = np.zeros(n_steps)
        e5 = np.zeros(n_steps)
        z_surface = np.zeros(n_steps)
        
        for i in range(n_steps):
            ee_x = ee_task[0, i]
            ee_y = ee_task[1, i]
            ee_z = ee_task[2, i]
            
            ee_vy = self.simulation_model.ee_velocity(i)[1]
            n = self.surface.get_normal_vector_casadi()(ee_x, ee_y)
            ee_rot_vector = self.simulation_model.ee_orientation(i)

            rot_z = np.array([0.0, 0.0, ee_rot_vector[2]])
            rot_y = np.array([0.0, ee_rot_vector[1], 0.0])
            
            z_surf = self.surface.get_point_on_surface(ee_x, ee_y)
            e1[i] = z_surf - ee_z
            e2[i] = 1 - (n.T @ rot_z)
            e3[i] = np.array([1.0, 0.0, 0.0]).T @ rot_y
            e4[i] = self.px_ref - ee_x
            e5[i] = self.vy_ref - ee_vy
            z_surface[i] = z_surf
        
        return {
            'e1': e1,
            'e2': e2,
            'e3': e3,
            'e4': e4,
            'e5': e5,
            'z_surface': z_surface
        }
    
    @cached_property
    def metrics(self):
        """
        Compute all performance metrics in one pass (cached).
        
        Returns:
            dict with keys:
            - rmse: dict with e1, e2, e3, e4
            - itse: dict with e1, e2, e3, e4
            - max_combined_rmse: maximum combined error across all constraints
        """
        if not self._data_computed:
            raise RuntimeError("Must call run() before accessing metrics")
        
        errors = self.errors
        
        # Calculate RMSE for all errors
        rmse = {}
        for key in ['e1', 'e2', 'e3', 'e4']:
            error_signal = errors[key]
            rmse[key] = np.sqrt(np.sum(error_signal**2) / self.simulation_time)
        
        # Calculate ITSE for all errors
        itse = {}
        for key in ['e1', 'e2', 'e3', 'e4']:
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
            'itse_e1': m['itse']['e1'],
            'itse_e2': m['itse']['e2'],
            'itse_e3': m['itse']['e3'],
            'itse_e4': m['itse']['e4'],
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
                    name = name_template(dict(zip(param_paths, values)), coeff_idx)
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
        'simulation_time': 1.0,
        'prediction_horizon': 200,
        'surface_limits': ((-2, 2), (-2, 2)),
        'surface_origin': np.array([0.0, 0.0, 0.0]),
        'surface_orientation_rpy': np.array([0.0, 0.0, 0.0]),
        'q_0': np.array([np.pi/3, -np.pi/3, np.pi/4, -np.pi/2, -np.pi/2, 0.0]),
        'qdot_0': np.array([2, 2, 2, 2, 2, 2]),
        'wcv': np.array([5, 10, 15, 20, 25, 35]),
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