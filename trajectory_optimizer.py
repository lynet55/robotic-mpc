import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from scipy.linalg import block_diag

class MPC:
    
    def __init__(self, surface, state, initial_state, control_input, model, 
                 forward_kinematics, differential_kinematics, N_horizon, Tf, 
                 px_ref= 0.4, vy_ref=-0.40, surface_position=None, surface_rpy=None, 
                 translation_ee_t=[0,0,0.40]):
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

        self._surface_expression = surface.get_surface_function() # CasADi expression for the target surface
        self._state = state
        self.surface = surface # Surface object 
        self.surface_pos_world = surface.get_position() # Surface frame origin with respect world
        self.surface_rpy_world = surface.get_orientation_rpy() # Surface frame orientation with respect world
        self.N_horizon = N_horizon
        self.Tf = Tf

        self.px_ref = px_ref
        self.vy_ref = vy_ref

        if isinstance(translation_ee_t, (list, np.ndarray)):
            self.translation = ca.vertcat(translation_ee_t[0], 
                                          translation_ee_t[1], 
                                          translation_ee_t[2])
        else:
            self.translation = translation_ee_t
        
        # Create ocp object to formulate the OCP
        self.ocp = AcadosOcp()

        # SET OCP OPTIONS
        self.ocp.solver_options.nlp_solver_type = 'SQP' # [SQP, 'SQP_RTI', 'DDP','SQP_WITH_FEASIBLE_QP']
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # ['GAUSS_NEWTON', 'EXACT']

        self.ocp.solver_options.print_level = 3 # verbosity of printing 

        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.qp_tol = 1e-8
        self.ocp.code_export_directory = 'c_generated_code_ocp'

        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.sim_method_num_stages = 4 # (deafult) RK4
        self.ocp.solver_options.sim_method_num_steps = 1

        # set prediction horizon
        self.ocp.solver_options.N_horizon = N_horizon
        self.ocp.solver_options.tf = Tf

        # MODEL 
        self.acados_model = model.acados_model
        self.y = model.y
        self.ocp.model =self.acados_model
        self.ocp.model.name = 'six_dof_robot_model'

        self.nx = self.acados_model.x.rows()
        self.nu = self.acados_model.u.rows()
        self.ny = self.nx + self.nu

        # OUTPUT 
        n_dof = self.nu 
        #q = self.acados_model.x[:n_dof]
        #q_dot = self.acados_model.x[n_dof:]
               
        #pose_ee_rot = forward_kinematics(q)

        #vee = differential_kinematics(q, q_dot)

        pose_ee_rot = self.y[:12]
        vee = self.y[12:]

        vee_x = vee[0]
        vee_y = vee[1]


        # From end-effector to task frame
        p_task, R_task = self._ee_to_task_transform(pose_ee_rot)

        # Task frame origin
        p_task_x = p_task[0]
        p_task_y = p_task[1]
        p_task_z = p_task[2]

        R_task_x = R_task[:3]
        # Task frame y axis
        R_task_y = R_task[3:6]
        # Task frame z axis
        R_task_z = R_task[6:]

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
        g3 = R_task_x[1]
        g4 = p_task_y
        g5 = vee_x

        g = ca.vertcat(g1, g2, g3, g4, g5)
        
        # Task errors weights
        w_origin_task = 100.0 
        w_normal_alignment_task = 50.0     
        w_x_alignment_task = 200.0
        w_fixed_x_task = 200.0  
        w_fixed_vy_task = 100.0

        # Control effort weight
        w_u = 0.01

        # Weights diagonal matrices
        Q = np.array([ w_origin_task,     
                       w_normal_alignment_task,        
                       w_x_alignment_task,            
                       w_fixed_x_task,       
                       w_fixed_vy_task      
                    ])
        R = 2 * np.array([w_u, w_u, w_u, w_u, w_u, w_u])

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

        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')

    
    def _ee_to_task_transform(self, pose_ee):
        """
        Map the end-effector pose in the world frame to the task frame pose
        in the world frame.

        The end-effector pose is given as a 12x1 CasADi vector
        [p_ee; R_w_ee(:)], where:
            - p_ee ∈ R^3 is the position of the EE origin expressed in WORLD
            - R_w_ee ∈ R^{3x3} is the rotation matrix from EE frame to WORLD,
            stored row-wise as:
                [R11, R12, R13,
                R21, R22, R23,
                R31, R32, R33]^T

        The task frame is defined as a frame rigidly attached to the EE frame,
        obtained by:
            - a rotation of π around the EE x-axis
            - a translation t expressed in the EE frame

        Args:
            pose_ee: CasADi SX/DX vector of size 12:
                    [p_ee(0:3); R_w_ee flattened row-wise (9 elements)].

        Returns:
            p_t:         3x1 CasADi vector, position of the task frame origin
                        expressed in WORLD coordinates.
            R_w_t_flat:  9x1 CasADi vector, rotation matrix R_w_t flattened
                        row-wise (same convention as pose_ee).
        """

        p_ee = pose_ee[:3]

        R_w_ee = ca.vertcat(
            ca.hcat([pose_ee[3],  pose_ee[4],  pose_ee[5]]),
            ca.hcat([pose_ee[6],  pose_ee[7],  pose_ee[8]]),
            ca.hcat([pose_ee[9],  pose_ee[10], pose_ee[11]]),
        )

        c = -1
        s = 0
        R_ee_t = ca.vertcat(
            ca.hcat([1, 0,  0]),
            ca.hcat([0, 1, 0]),
            ca.hcat([0, 0,  1]),
        )

        R_w_t = R_w_ee @ R_ee_t

        p_t = p_ee + R_w_ee @ self.translation

        R_w_t_flat = ca.vertcat(
            R_w_t[0, 0], R_w_t[1, 0], R_w_t[2, 0],
            R_w_t[0, 1], R_w_t[1, 1], R_w_t[2, 1],
            R_w_t[0, 2], R_w_t[1, 2], R_w_t[2, 2],
        )

        return p_t, R_w_t_flat



    def setup_integrator(self, dt):
        sim = AcadosSim()
        sim.model = self.acados_model

        sim.solver_options.T = dt # simulation time
        sim.solver_options.num_steps = 2 # Make extra integrator more precise than ocp-internal integrator
        sim.code_export_directory = 'c_generated_code_sim'
        acados_integrator = AcadosSimSolver(sim)
        return acados_integrator

    