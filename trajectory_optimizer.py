import casadi as ca
import numpy as np
import uuid
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim

class MPC:
    
    def __init__(self, surface, initial_state, model, N_horizon, Tf, 
                 px_ref= 0.40, vy_ref=-0.40):
        """
        Initialize MPC controller with explicit dependencies.
        
        Args:
            surface: CasADi expression for the target surface
            state: CasADi symbolic state variable [q, q_dot]
            control_input: CasADi symbolic control variable
            model: (Class prediction_model) 
            forward_kinematics: CasADi Function that maps q -> [px, py, pz, R11, R12, R13, R21, R22, R23, R31, R32, R33]
            differential_kinematics: CasADi Function that maps (q, q_dot) -> [vx, vy, vz, wx, wy, wz]
            surface_position: 3D position of surface origin [x, y, z]
            surface_rpy: Surface orientation as [roll, pitch, yaw] in radians
            translation_ee_t: Translation from EE origin to task origin, expressed in the
                              EE frame. Can be a length-3 iterable or a 3x1 CasADi vector.
        """

        self.surface = surface 
        self.surface_pos_world = surface.get_position() 
        self.surface_rpy_world = surface.get_orientation_rpy() 
        self.N_horizon = N_horizon
        self.Tf = Tf

        self.px_ref = px_ref
        self.vy_ref = vy_ref

        # Generate unique identifier for this MPC instance to avoid acados caching issues
        self._instance_id = uuid.uuid4().hex[:8]

        # Task errors weights
        self.w_origin_task = 200.0 
        self.w_normal_alignment_task = 50.0     
        self.w_x_alignment_task = 300.0
        self.w_fixed_x_task = 200.0  
        self.w_fixed_vy_task = 100.0

        # Control effort weight
        self.w_u = 0.01

        # Create ocp object to formulate the OCP
        self.ocp = AcadosOcp()

        # SET OCP OPTIONS
        self.ocp.solver_options.nlp_solver_type = 'SQP' # [SQP, 'SQP_RTI', 'DDP','SQP_WITH_FEASIBLE_QP']
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # ['GAUSS_NEWTON', 'EXACT']

        self.ocp.solver_options.print_level = 0 

        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.qp_tol = 1e-8
        # Use unique directory per instance to prevent caching conflicts
        self.ocp.code_export_directory = f'c_generated_code_ocp_{self._instance_id}'

        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.sim_method_num_stages = 4
        self.ocp.solver_options.sim_method_num_steps = 1

        # set prediction horizon
        self.ocp.solver_options.N_horizon = N_horizon
        self.ocp.solver_options.tf = Tf

        # MODEL 
        self.acados_model = model.acados_model
        self.y = model.y
        self.ocp.model =self.acados_model
        # Use unique model name per instance to prevent acados from reusing cached solver
        self.ocp.model.name = f'six_dof_robot_{self._instance_id}'

        self.nx = self.acados_model.x.rows()
        self.nu = self.acados_model.u.rows()
        self.ny = self.nx + self.nu

        # OUTPUT 
        n_dof = self.nu 
        p_task = self.y[0:3]
        p_task_x = p_task[0]
        p_task_y = p_task[1]
        p_task_z = p_task[2]


        R_task_x = self.y[3:6]
        R_task_y = self.y[6:9]
        R_task_z = self.y[9:12]

        vee = self.y[12:18]
        v_t_x = vee[0]
        v_t_y = vee[1]
        v_t_z = vee[2]

        # Normal versor of the surface 
        n_fun = self.surface.get_normal_vector_casadi()  
        n = n_fun(p_task_x, p_task_y)
        
        # COST
        # Definition of the reference output function g_ref, used to express task constraints
        g1_ref = 0     
        g2_ref = 1  
        g3_ref = 0
        g4_ref = px_ref  
        g5_ref = vy_ref

        g_ref = [g1_ref, g2_ref, g3_ref, g4_ref, g5_ref]

        # Definition of task constraints
        S = self.surface.get_surface_function()

        g1 = S(p_task_x,p_task_y) - p_task_z
        g2 = ca.dot(n, R_task_z)
        g3 = R_task_y[0]
        g4 = p_task_x
        g5 = v_t_y

        g = ca.vertcat(g1, g2, g3, g4, g5)
        
        # Weights diagonal matrices
        Q = np.array([ self.w_origin_task,     
                       self.w_normal_alignment_task,        
                       self.w_x_alignment_task,            
                       self.w_fixed_x_task,       
                       self.w_fixed_vy_task      
                    ])
        R = 2 * np.array([self.w_u, self.w_u, self.w_u, self.w_u, self.w_u, self.w_u])

        W = np.diag(np.concatenate([Q, R]))

        y = ca.vertcat(g, self.acados_model.u)
        y_ref = np.concatenate([g_ref, np.zeros(self.nu)])

        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.model.cost_y_expr = y
        self.ocp.cost.yref = y_ref
        self.ocp.cost.W = W

        # Control input bounds (joint velocity commands)
        q_dot_ref_max = 500.0  # Maximum joint velocity command

        # Control input bounds (joint velocity commands)
        self.ocp.constraints.lbu = np.array([-q_dot_ref_max] * 6)
        self.ocp.constraints.ubu = np.array([q_dot_ref_max] * 6)
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5])

        # Initial state constraint will be set at runtime
        self.ocp.constraints.x0 = initial_state
        # Use unique JSON file per instance
        self.solver = AcadosOcpSolver(self.ocp, json_file=f'acados_ocp_{self._instance_id}.json')