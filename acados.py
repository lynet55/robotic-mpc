import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

class MPC:
    
    def __init__(self, surface, state, initial_state, control_input, dynamics, 
                 forward_kinematics, differential_kinematics,
                 surface_position=None, surface_orientation_rpy=None, 
                 desired_offset=1.0):
        """
        Initialize MPC controller with explicit dependencies.
        
        Args:
            surface: CasADi expression for the target surface
            state: CasADi symbolic state variable [q, q_dot]
            control_input: CasADi symbolic control variable
            dynamics: CasADi expression for state derivative (dx/dt)
            forward_kinematics: CasADi Function that maps q -> [px, py, pz, qx, qy, qz, qw]
            differential_kinematics: CasADi Function that maps (q, q_dot) -> [vx, vy, vz, wx, wy, wz]
            surface_position: 3D position of surface origin [x, y, z]
            surface_orientation_rpy: Surface orientation as [roll, pitch, yaw] in radians
            desired_offset: Desired offset distance from surface (default: 1.0)
        """
        self._surface_expression = surface.get_quadratic_surface()
        self._state = state
        self.surface_position = surface.get_position()
        self.surface_orientation_rpy = surface.get_orientation_rpy()
        self.desired_offset = surface.get_desired_offset()

        self.ocp = AcadosOcp()
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hpipm_mode = 'BALANCE'
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.nlp_solver_type = 'SQP'
        self.ocp.solver_options.qp_solver_iter_max = 50
        self.ocp.solver_options.nlp_solver_max_iter = 100
        self.ocp.solver_options.tf = 2.0

        self.ocp.dims.N = 20

        self.model = self.ocp.model
        self.model.name = 'six_dof_robot_model'
        
        # Setup model variables and dynamics
        n_states = state.shape[0]
        x_dot = ca.SX.sym('x_dot', n_states)
        
        self.model.x = state  # State is [q, q_dot] - joint positions and velocities
        self.model.xdot = x_dot  # State derivative
        self.model.u = control_input
        self.model.f_expl_expr = dynamics
        self.model.f_impl_expr = x_dot - dynamics

        # Compute end-effector quantities from state
        n_dof = n_states // 2
        q = state[:n_dof]
        q_dot = state[n_dof:]
        
        # Call the kinematics functions with the state
        ee_pose = forward_kinematics(q)  # Returns [px, py, pz, qx, qy, qz, qw]
        ee_vel = differential_kinematics(q, q_dot)  # Returns [vx, vy, vz, wx, wy, wz]

        
        # Extract Cartesian position and velocity components (world frame)
        px_world = ee_pose[0]
        py_world = ee_pose[1]
        pz_world = ee_pose[2]
        
        vx = ee_vel[0]
        vy = ee_vel[1]
        vz = ee_vel[2]
        
        # Create rotation matrix from RPY (Roll-Pitch-Yaw) for surface frame
        roll = self.surface_orientation_rpy[0]
        pitch = self.surface_orientation_rpy[1]
        yaw = self.surface_orientation_rpy[2]
        
        # Rotation matrices for each axis
        Rx = ca.vertcat(
            ca.horzcat(1, 0, 0),
            ca.horzcat(0, ca.cos(roll), -ca.sin(roll)),
            ca.horzcat(0, ca.sin(roll), ca.cos(roll))
        )
        
        Ry = ca.vertcat(
            ca.horzcat(ca.cos(pitch), 0, ca.sin(pitch)),
            ca.horzcat(0, 1, 0),
            ca.horzcat(-ca.sin(pitch), 0, ca.cos(pitch))
        )
        
        Rz = ca.vertcat(
            ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0),
            ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0),
            ca.horzcat(0, 0, 1)
        )
        
        # Combined rotation matrix: R = Rz * Ry * Rx
        R_world_to_surf = Rz @ Ry @ Rx
        
        # Transform EE position from world frame to surface frame
        ee_pos_world = ca.vertcat(px_world, py_world, pz_world)
        surf_pos = ca.vertcat(self.surface_position[0], self.surface_position[1], self.surface_position[2])
        
        # Position in surface frame = R^T * (p_world - p_surface)
        ee_pos_surf = R_world_to_surf.T @ (ee_pos_world - surf_pos)

        
        # Extract surface frame coordinates
        px_surf = ee_pos_surf[0]  # x in surface frame
        py_surf = ee_pos_surf[1]  # y in surface frame
        pz_surf = ee_pos_surf[2]  # z in surface frame (perpendicular to surface)
        
        # Evaluate surface height at current x,y position in surface frame
        # The surface expression should use px_surf and py_surf
        surface_height = ca.substitute(self._surface_expression, 
                                        ca.vertcat(ca.SX.sym("x"), ca.SX.sym("y")),
                                        ca.vertcat(px_surf, py_surf))

        # Cost function weights
        w_position_xy = 5.0     # Weight for tracking surface in x-y
        w_position_z = 10.0     # Weight for maintaining desired offset
        w_velocity = 0.1
        w_orientation = 1.0     # Weight for perpendicular orientation

        # Surface tracking errors
        # Error in z-direction: maintain offset of desired_offset above surface
        z_error = pz_surf - (surface_height + self.desired_offset)
        

        # Cost functions
        running_cost = w_position_z * z_error**2
        running_cost += w_velocity * (vx**2 + vy**2 + vz**2)
        
        terminal_cost = w_position_z * z_error**2
        terminal_cost += w_velocity * (vx**2 + vy**2 + vz**2)

        self.model.cost_expr_ext_cost = running_cost
        self.model.cost_expr_ext_cost_e = terminal_cost

        # Control input bounds (joint velocity commands)
        self.ocp.constraints.lbu = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0])
        self.ocp.constraints.ubu = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5])

        # Initial state (12 dimensions for 6-DOF robot: [q(6), q_dot(6)])
        self.x0 = initial_state  # All joints at zero position and velocity
        self.ocp.constraints.x0 = initial_state
        
        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')