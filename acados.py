import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from scipy.linalg import block_diag

class MPC:
    
    def __init__(self, surface, state, initial_state, control_input, dynamics, 
                 forward_kinematics, differential_kinematics,
                 surface_position=None, surface_rpy=None, 
                 desired_offset=1.0):
        """
        Initialize MPC controller with explicit dependencies.
        
        Args:
            surface: CasADi expression for the target surface
            state: CasADi symbolic state variable [q, q_dot]
            control_input: CasADi symbolic control variable
            dynamics: CasADi expression for state derivative (dx/dt)
            forward_kinematics: CasADi Function that maps q -> [px, py, pz, R11, R12, R13, R21, R22, R23, R31, R32, R33]
            differential_kinematics: CasADi Function that maps (q, q_dot) -> [vx, vy, vz, wx, wy, wz]
            surface_position: 3D position of surface origin [x, y, z]
            surface_rpy: Surface orientation as [roll, pitch, yaw] in radians
            desired_offset: Desired offset distance from surface (default: 1.0)
        """
        self._surface_expression = surface.get_quadratic_surface()
        self._state = state
        self.surface = surface
        self.surface_pos_world = surface.get_position()
        self.surface_rpy_world = surface.get_orientation_rpy()
        self.desired_offset = desired_offset

        self.ocp = AcadosOcp()
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hpipm_mode = 'BALANCE'
        self.ocp.solver_options.integrator_type = 'ERK'
        # self.ocp.solver_options.nlp_solver_type = 'SQP'
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp.solver_options.qp_solver_warm_start = 1
        self.ocp.solver_options.qp_solver_iter_max = 50
        self.ocp.solver_options.nlp_solver_max_iter = 50
        self.ocp.solver_options.tf = 1.5
        self.ocp.solver_options.qp_solver_cond_N = 5
        self.ocp.dims.N = 15

        self.ocp.model.name = 'six_dof_robot_model'

        self.nx = state.shape[0]
        self.nu = control_input.shape[0]

        x_dot = ca.SX.sym('x_dot', self.nx)
        p = ca.SX.sym('p', 6)
        
        self.ocp.model.x = state  # State is [q, q_dot] - joint positions and velocities
        self.ocp.model.xdot = x_dot
        self.ocp.model.u = control_input
        self.ocp.model.f_expl_expr = dynamics
        self.ocp.model.f_impl_expr = x_dot - dynamics
        self.ocp.model.p = p

        # Compute end-effector quantities from state
        n_dof = self.nx // 2
        q = state[:n_dof]
        q_dot = state[n_dof:]
        
        # Call the kinematics functions with the state       
        fk_out = forward_kinematics(q)
        ee_x_world = fk_out[0]
        ee_y_world = fk_out[1]
        ee_z_world = fk_out[2]
        # Extract rotation matrix elements [R11, R12, R13, R21, R22, R23, R31, R32, R33]
        R11 = fk_out[3]
        R12 = fk_out[4]
        R13 = fk_out[5]
        R21 = fk_out[6]
        R22 = fk_out[7]
        R23 = fk_out[8]
        R31 = fk_out[9]
        R32 = fk_out[10]
        R33 = fk_out[11]
        
        dk_out = differential_kinematics(q, q_dot)
        ee_vx_world = dk_out[0]
        ee_vy_world = dk_out[1]
        ee_vz_world = dk_out[2]
        ee_wx_world = dk_out[3]
        ee_wy_world = dk_out[4]
        ee_wz_world = dk_out[5]

        ee_pos_world = ca.vertcat(ee_x_world, ee_y_world, ee_z_world)
        surface_pos_world = ca.vertcat(self.surface_pos_world[0], self.surface_pos_world[1], self.surface_pos_world[2])
        surface_rpy_world = ca.vertcat(self.surface_rpy_world[0], self.surface_rpy_world[1], self.surface_rpy_world[2])

        # Transform EE position to surface frame
        ee_pos_surf = self._ee_to_surface_transform_2(ee_pos_world)
        
        # Define parameters: [x_ref, y_ref, z_ref, nx, ny, nz]
        # where (x_ref, y_ref, z_ref) is reference position in surface frame
        # and (nx, ny, nz) is the surface normal vector
        # Extract z-axis (third column) from rotation matrix
        ee_z_axis = ca.vertcat(R13, R23, R33)

        x_ref = p[0]
        y_ref = p[1]
        z_ref = p[2]
        normal_ref = ca.vertcat(p[3], p[4], p[5])
        normal_alignment = ca.dot(ee_z_axis, normal_ref)

        # Define output function for cost
        v_ref = ca.DM([0.2, 0.0, 0.0])      # constant desired EE linear velocity
        u_ref = ca.DM([0, 0, 0, 0, 0, 0])   # constant desired control (usually zeros)

        h = ca.vertcat(
            ee_pos_surf[0] - x_ref,
            ee_pos_surf[1] - y_ref,
            ee_pos_surf[2] - z_ref,
            # ee_vx_world - v_ref[0],
            # ee_vy_world - v_ref[1],
            # ee_vz_world - v_ref[2],
            # 1.0 - normal_alignment,
            control_input - u_ref
        )
        
        
        # Cost function weights
        w_task_xy_surface = 100.0 
        w_task_z_surface = 10.0     
        w_task_velocity_xyz_surface = 0.1
        # w_task_normal_surface = 100.0  # High weight for orientation alignment
        w_control_input = 0.01
        
        W = np.diag([
            w_task_xy_surface,           # x position
            w_task_xy_surface,           # y position
            w_task_z_surface,            # z position (offset)
            # w_task_velocity_xyz_surface, # vx
            # w_task_velocity_xyz_surface, # vy
            # w_task_velocity_xyz_surface, # vz
            # w_task_normal_surface,       # alignment error
            w_control_input,             # u[0]
            w_control_input,             # u[1]
            w_control_input,             # u[2]
            w_control_input,             # u[3]
            w_control_input,             # u[4]
            w_control_input              # u[5]
        ])


        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.model.cost_y_expr = h
        self.ocp.cost.W = W
        self.ocp.cost.yref = np.zeros(W.shape[0])
        self.ocp.cost.yref_0 = np.zeros(W.shape[0])

        # self.ocp.model.cost_expr_ext_cost = 0.5 * h.T @ W @ h
        
        # Set up parameters
        self.ocp.dims.np = 6
        self.ocp.parameter_values = np.zeros(6)
        
        # Terminal cost (optional - higher weights on position)
        h_e = ca.vertcat(
            ee_pos_surf[0] - x_ref,
            ee_pos_surf[1] - y_ref,
            ee_pos_surf[2] - z_ref,
            1.0 - normal_alignment
        )
        
        W_e = np.diag([50.0, 50.0, 100.0, 200.0])  # Higher terminal weights
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        self.ocp.model.cost_y_expr_e = h_e
        self.ocp.cost.W_e = W_e
        self.ocp.cost.yref_e = np.zeros(W_e.shape[0])

        # Control input bounds (joint velocity commands)
        self.ocp.constraints.lbu = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0])
        self.ocp.constraints.ubu = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5])

        # Initial state constraint will be set at runtime
        self.ocp.constraints.x0 = initial_state

        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')

    def _ee_to_surface_transform(self, point_world, surface_pos_world, surface_rpy_world):
        """
        Transform a point from world frame to surface frame.
        
        Args:
            point_world: CasADi expression for point in world frame [x, y, z]
            surface_pos_world: Surface position in world frame
            surface_rpy_world: Surface orientation (roll, pitch, yaw)
        
        Returns:
            CasADi expression for point in surface frame
        """
        # Translation: move to surface origin
        point_translated = point_world - surface_pos_world + np.array([0.0, 0.0, 0.1])
        
        # Rotation from world to surface frame (inverse of surface orientation)
        # For simplicity, assuming surface frame aligned with world
        # You should implement proper rotation based on surface_rpy_world
        
        # Placeholder: just translation for now
        # TODO: Add proper rotation matrix based on RPY
        transformed_point = point_translated
        
        return transformed_point
    
    def _ee_to_surface_transform_2(self, point):
        """
        Apply homogeneous transform to a CasADi expression:
        - Rotation of π around x-axis
        - Translation of 1 unit in z-direction
        
        Args:
            point: CasADi expression representing a 3D point [x, y, z]
        
        Returns:
            CasADi expression representing the transformed point
        """
   
        s = ca.sin(np.pi)
        c = ca.cos(np.pi)
         # Translation vector: 0.3 unit in z-direction
        translation = ca.vertcat(0, 0, 0.1)
         
         # Build homogeneous transform H = [R t; 0 0 0 1]
        H = ca.vertcat(
             ca.horzcat(1, 0, 0, translation[0]),
             ca.horzcat(0, c, -s, translation[1]),
             ca.horzcat(0, s,  c, translation[2]),
             ca.horzcat(0, 0, 0, 1)
         )
         
         # Apply homogeneous transform to point: p' = H * [p; 1]
        point_h = ca.vertcat(point, 1)
        transformed_point_h = H @ point_h
        transformed_point = ca.vertcat(
             transformed_point_h[0],
             transformed_point_h[1],
             transformed_point_h[2]
        )
         
        return transformed_point
    

    