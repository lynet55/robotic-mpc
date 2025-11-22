import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

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
            forward_kinematics: CasADi Function that maps q -> [px, py, pz, qx, qy, qz, qw]
            differential_kinematics: CasADi Function that maps (q, q_dot) -> [vx, vy, vz, wx, wy, wz]
            surface_position: 3D position of surface origin [x, y, z]
            surface_rpy: Surface orientation as [roll, pitch, yaw] in radians
            desired_offset: Desired offset distance from surface (default: 1.0)
        """
        self._surface_expression = surface.get_quadratic_surface()
        self._state = state
        self.surface_pos_world = surface.get_position()
        self.surface_rpy_world = surface.get_orientation_rpy()
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
        ee_x_world, ee_y_world, ee_z_world, qx, qy, qz, qw = forward_kinematics(q)
        ee_vx_world, ee_vy_world, ee_vz_world, ee_wx_world, ee_wy_world, ee_wz_world = differential_kinematics(q, q_dot)  # Returns [vx, vy, vz, wx, wy, wz]

        ee_pos_world = ca.vertcat(ee_x_world, ee_y_world, ee_z_world)
        surface_pos_world = ca.vertcat(self.surface_pos_world[0], self.surface_pos_world[1], self.surface_pos_world[2])
        surface_rpy_world = ca.vertcat(self.surface_rpy_world[0], self.surface_rpy_world[1], self.surface_rpy_world[2])

        ee_pos_surf, ee_rpy_surf  = self._ee_to_surface(ee_pos_world, surf_pos_world, surface_rpy_world)
        
        # Evaluate surface height at current x,y position in surface frame
        # The surface expression should use px_surf and py_surf
        # surface_height = ca.substitute(self._surface_expression, 
        #                                 ca.vertcat(ca.SX.sym("x"), ca.SX.sym("y")),
        #                                 ca.vertcat(p, py_surf))

        # Cost function weights
        w_position_xy = 5.0     # Weight for tracking surface in x-y
        w_position_z = 10.0     # Weight for maintaining desired offset
        w_velocity = 0.1
        w_orientation = 1.0     # Weight for perpendicular orientation

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
        self.x0 = initial_state
        self.ocp.constraints.x0 = initial_state
        
        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')

    def ee_to_surface(self, ee_pose_world):
