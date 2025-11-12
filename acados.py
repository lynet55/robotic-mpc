import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

class MPC:
    
    def __init__(self, surface, state, control_input, dynamics, 
                 forward_kinematics, differential_kinematics):
        """
        Initialize MPC controller with explicit dependencies.
        
        Args:
            surface: CasADi expression for the target surface
            state: CasADi symbolic state variable [q, q_dot]
            control_input: CasADi symbolic control variable
            dynamics: CasADi expression for state derivative (dx/dt)
            forward_kinematics: CasADi Function that maps q -> [px, py, pz, qx, qy, qz, qw]
            differential_kinematics: CasADi Function that maps (q, q_dot) -> [vx, vy, vz, wx, wy, wz]
        """
        self._surface = surface
        self._state = state

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
        
        # Extract Cartesian position and velocity components
        px = ee_pose[0]
        py = ee_pose[1]
        pz = ee_pose[2]
        
        vx = ee_vel[0]
        vy = ee_vel[1]
        vz = ee_vel[2]

        # Cost function weights
        w_position = 10.0
        w_velocity = 0.1

        # Surface tracking error
        surface_error_z = pz - surface  # You may want to substitute px, py into surface

        # Cost functions
        running_cost = w_position * surface_error_z**2
        running_cost += w_velocity * (vx**2 + vy**2 + vz**2)
        
        terminal_cost = w_position * surface_error_z**2
        terminal_cost += w_velocity * (vx**2 + vy**2 + vz**2)

        self.model.cost_expr_ext_cost = running_cost
        self.model.cost_expr_ext_cost_e = terminal_cost

        # Control input bounds (joint velocity commands)
        self.ocp.constraints.lbu = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0])
        self.ocp.constraints.ubu = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5])

        # Initial state (12 dimensions for 6-DOF robot: [q(6), q_dot(6)])
        self.x0 = np.zeros(12)  # All joints at zero position and velocity
        self.ocp.constraints.x0 = self.x0
        
        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')