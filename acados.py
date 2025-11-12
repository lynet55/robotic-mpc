import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

class Acados:
    
    def _init_(self, surface, state, x_dot, control_inputs, explicit_model, implisit_model):
        self._surface = surface
        self._state = state
        self._x_dot = x_dot

        self.opc = AcadosOcp()
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
        self.model.x = state
        self.model.xdot = x_dot
        self.model.u = control_inputs
        self.model.f_expl_expr = explicit_model
        self.model.f_impl_expr = implisit_model

        w_position = 10.0
        w_velocity = 0.1

        px = state[0]
        py = state[1]
        pz = state[2]
        vx = state[3]
        vy = state[4]
        vz = state[5]

        running_cost = 0.0
        terminal_cost = 0.0

        surface_error_z = pz - surface[px,py]

        running_cost += w_position * surface_error_z**2
        running_cost += w_velocity * (vx*2 + vy2 + vz*2)
        terminal_cost += w_position* surface_error_z**2
        terminal_cost += w_velocity * (vx*2 + vy2 + vz*2)

        self.model.cost_expr_ext_cost = running_cost
        self.model.cost_expr_ext_cost_e = terminal_cost

        self.ocp.constraints.lbu = np.array([-2.0, -2.0, -5.0])  # min acceleration
        self.ocp.constraints.ubu = np.array([2.0, 2.0, 5.0])     # max acceleration
        self.ocp.constraints.idxbu = np.array([0, 1, 2])

        self.ocp.constraints.lbu = np.array([-2.0, -2.0])
        self.ocp.constraints.ubu = np.array([2.0, 2.0])
        self.ocp.constraints.idxbu = np.array([0, 1])  # only u[0] and u[1]

        #         # State bounds (e.g., height must be positive)
        # self.ocp.constraints.Jbx = np.eye(6)[[2]]  # constrain z
        # self.ocp.constraints.lbx = np.array([0.0])
        # self.ocp.constraints.ubx = np.array([15.0])
        # self.ocp.constraints.idxbx = np.array([2])

        # h_expr = ca.vertcat(
        #     ca.sqrt(vx*2 + vy2 + vz*2)  # total velocity
        # )
        # self.model.con_h_expr = h_expr
        # self.ocp.constraints.lh = np.array([0.0])
        # self.ocp.constraints.uh = np.array([5.0])  # max speed 5 m/s

    
        self.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.ocp.constraints.x0 = self.x0
        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')