import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin
from acados_template import AcadosModel


class SixDofRobot:
    def __init__(self, urdf_loader, Ts, Wcv, translation_ee_t=[0,0,0]):  
        # Numeric Pinocchio model (for loading URDF and reference) |Deprecable?|
        self._model = urdf_loader.model
        # CasADi Pinocchio model (for symbolic computations)
        self._cmodel = cpin.Model(self._model)
        self._cdata = self._cmodel.createData()
        
        # Get the number of degrees of freedom
        self._n_dof = self._cmodel.nq

        if isinstance(translation_ee_t, (list, np.ndarray)):
            self.translation = ca.vertcat(translation_ee_t[0], 
                                          translation_ee_t[1], 
                                          translation_ee_t[2])
        else:
            self.translation = translation_ee_t
        
        # Get end-effector frame ID from both models
        # Use 'tool0' as the end-effector frame (standard UR5 frame)
       
        self._cee_frame_id = self._cmodel.getFrameId('tool0')
        
        # Check if frame ID is valid
        if self._cee_frame_id >= self._model.nframes:
            raise ValueError(f"Frame 'tool0' not found in model. Available frames: {[self._model.frames[i].name for i in range(self._model.nframes)]}")
        
        # Speed loop bandwidths (diagonal matrix)
        if Wcv is None:
            Wcv = np.ones(self._n_dof)
            print("Wcv not specified: default parameters used [1,...,1]")  
        self._Wcv = Wcv
        
        self._Ts = Ts

        # Define CasADi symbolic variables
        self.state = ca.SX.sym('x', 2 * self._n_dof)  # [q, q_dot]
        self.input = ca.SX.sym('u', self._n_dof)    # control input (velocity commands)
        
        # Create CasADi functions for Pinocchio operations
        self._setup_casadi_functions()
        
        # Define the dynamics as an ODE
        #self.acados_model, self.y = self._generate_dynamics_model()

        self._Ad = None   
        self._Bd = None  
        self._update_discrete_lti_matrices()

        self.acados_model, self.y = self._generate_dynamics_model()

    @property
    def Ad(self):
        if self._Ad is None:
            raise RuntimeError("Ad not initialized")
        return self._Ad

    @property
    def Bd(self):
        if self._Bd is None:
            raise RuntimeError("Bd not initialized")
        return self._Bd

    @property
    def Ts(self):
        return self._Ts
    
    @Ts.setter
    def Ts(self, new_Ts):
        if new_Ts <= 0:
            raise ValueError("The sampling time must be positive")
        if not isinstance(new_Ts, float):
            raise TypeError(f"Expected type float, got {type(new_Ts).__name__}.")
        self._Ts = new_Ts

    @property
    def n_dof(self):
        return self._n_dof
    

    def _update_discrete_lti_matrices(self,):
       
        nd = self._n_dof
        w = self._Wcv 
        Ts = self._Ts

        a22 = np.exp(-w * Ts)             
        a12 = (1.0 - a22) / w              

        I6 = np.eye(nd, dtype=np.float64)

        A12 = np.diag(a12)               
        A22 = np.diag(a22)                 

        B2  = np.diag(1.0 - a22)           
        B1  = Ts * I6 - A12              

        Ad = np.zeros((2*nd, 2*nd), dtype=np.float64)
        Bd = np.zeros((2*nd,  nd), dtype=np.float64)

        Ad[0:nd, 0:nd]   = I6
        Ad[0:nd, nd:2*nd]  = A12
        Ad[nd:2*nd, nd:2*nd] = A22

        Bd[0:nd, :]  = B1
        Bd[nd:2*nd, :] = B2

        self._Ad = Ad
        self._Bd = Bd

    def _setup_casadi_functions(self):
        """Setup pure CasADi functions for Pinocchio operations"""
        # Create symbolic variables for Pinocchio functions
        q_sym = ca.SX.sym('q', self._n_dof)
        q_dot_sym = ca.SX.sym('q_dot', self._n_dof)
        
        # Forward kinematics using CasADi Pinocchio (also updates joint placements)
        
        # Update all frame placements (including end-effector)
        cpin.forwardKinematics(self._cmodel, self._cdata, q_sym)
        cpin.updateFramePlacements(self._cmodel, self._cdata)
        
        # Get end-effector frame placement from data (use CasADi model frame ID)
        ee_transform = self._cdata.oMf[self._cee_frame_id]
        ee_position = ee_transform.translation  # 3D position (CasADi expression)
        ee_rotation = ee_transform.rotation     # 3x3 rotation matrix (CasADi expression)
        
        # Flatten rotation matrix to vector [R11, R12, R13, R21, R22, R23, R31, R32, R33]
        ee_rotation_flat = ca.vertcat(
            ee_rotation[0, 0], ee_rotation[0, 1], ee_rotation[0, 2],
            ee_rotation[1, 0], ee_rotation[1, 1], ee_rotation[1, 2],
            ee_rotation[2, 0], ee_rotation[2, 1], ee_rotation[2, 2]
        )
        
        # Concatenate position and rotation matrix [x, y, z, R11, R12, ..., R33]
        ee_pose_rotmat = ca.vertcat(ee_position, ee_rotation_flat)
        
        # Create FK function with rotation matrix output
        self.fk_casadi_rot = ca.Function('forward_kinematics',
                                      [q_sym],
                                      [ee_pose_rotmat],
                                      ['q'], ['pose'])
        
        ''' |Deprecable? No need of euler angles (unless visualizatiomn?)|
        # Convert rotation matrix to euler angles (ZYX convention)
        euler_angles = self._rotation_matrix_to_euler_casadi(ee_rotation)

        # Concatenate position and euler angles [x, y, z, roll, pitch, yaw]
        ee_pose_euler = ca.vertcat(ee_position, euler_angles)
        
        # Create FK function with euler angles output
        self.fk_casadi_euler = ca.Function('forward_kinematics_euler',
                                            [q_sym],
                                            [ee_pose_euler],
                                            ['q'], ['pose'])'''
        
        ee_quat = self._rot_to_quat_casadi(ee_rotation)  # [qx, qy, qz, qw]
        ee_pose_quat = ca.vertcat(ee_position, ee_quat)  
        self.fk_casadi_quat = ca.Function(
            'forward_kinematics_quat',
            [q_sym],
            [ee_pose_quat],
            ['q'], ['pose']
        )

        # Differential kinematics using CasADi Pinocchio
        J = cpin.computeFrameJacobian(self._cmodel, self._cdata, q_sym,
                                       self._cee_frame_id, pin.ReferenceFrame.WORLD)
        ee_velocity = J @ q_dot_sym
        
        self.dk_casadi = ca.Function('differential_kinematics',
                                      [q_sym, q_dot_sym],
                                      [ee_velocity],
                                      ['q', 'q_dot'], ['ee_vel'])
    
    def _rot_to_quat_casadi(self, R):
        eps = 1e-10
        trace = R[0,0] + R[1,1] + R[2,2]

        def branch_trace():
            w = 0.5 * ca.sqrt(ca.fmax(1 + trace, eps))
            x = (R[2,1] - R[1,2]) / (4*w + eps)
            y = (R[0,2] - R[2,0]) / (4*w + eps)
            z = (R[1,0] - R[0,1]) / (4*w + eps)
            return ca.vertcat(x,y,z,w)

        def branch_x():
            x = ca.sqrt(ca.fmax(1 + R[0,0] - R[1,1] - R[2,2], eps)) / 2
            w = (R[2,1] - R[1,2]) / (4*x + eps)
            y = (R[0,1] + R[1,0]) / (4*x + eps)
            z = (R[0,2] + R[2,0]) / (4*x + eps)
            return ca.vertcat(x,y,z,w)

        def branch_y():
            y = ca.sqrt(ca.fmax(1 - R[0,0] + R[1,1] - R[2,2], eps)) / 2
            w = (R[0,2] - R[2,0]) / (4*y + eps)
            x = (R[0,1] + R[1,0]) / (4*y + eps)
            z = (R[1,2] + R[2,1]) / (4*y + eps)
            return ca.vertcat(x,y,z,w)

        def branch_z():
            z = ca.sqrt(ca.fmax(1 - R[0,0] - R[1,1] + R[2,2], eps)) / 2
            w = (R[1,0] - R[0,1]) / (4*z + eps)
            x = (R[0,2] + R[2,0]) / (4*z + eps)
            y = (R[1,2] + R[2,1]) / (4*z + eps)
            return ca.vertcat(x,y,z,w)

        cond_trace = trace > 0
        cond_x = ca.logic_and(trace <= 0, ca.logic_and(R[0,0] > R[1,1], R[0,0] > R[2,2]))
        cond_y = ca.logic_and(trace <= 0, R[1,1] > R[2,2])

        quat = ca.if_else(cond_trace,
                        branch_trace(),
                        ca.if_else(cond_x,
                                    branch_x(),
                                    ca.if_else(cond_y,
                                                branch_y(),
                                                branch_z())))

        norm_q = ca.sqrt(ca.sumsqr(quat) + eps)
        return quat / norm_q

    ''' |Deprecable?
    def _rotation_matrix_to_euler_casadi(self, R):
        """
        Convert 3x3 rotation matrix to euler angles (ZYX convention) using CasADi
        
        Args:
            R: 3x3 CasADi rotation matrix
        
        Returns:
            euler angles [roll, pitch, yaw] as CasADi expression
        """
    
        pitch = ca.atan2(-R[2, 0], ca.sqrt(R[0, 0]**2 + R[1, 0]**2))
        roll = ca.atan2(R[2, 1], R[2, 2])
        yaw = ca.atan2(R[1, 0], R[0, 0])
        
        return ca.vertcat(roll, pitch, yaw)'''
    
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

        # Rotation of pi around x
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


    def _generate_dynamics_model(self) -> AcadosModel:
        """
        Create the ODE representation: dx/dt = f(x, u)
        
        Dynamics model:
        dq/dt = q_dot (integrator for the position)
        dq_dot/dt = -diag(th) @ q_dot + diag(th) @ u
        
        This is a first-order, decoupled, speed closed-loop model of the robot.
        """
        # States
        q = self.state[:self._n_dof]
        q_dot = self.state[self._n_dof:]

        # Output 
        pose_ee_rot  = self.fk_casadi_rot(q)  
        vee = self.dk_casadi(q, q_dot)

        p_task, R_task = self._ee_to_task_transform(pose_ee_rot)
        y = ca.vertcat(p_task, R_task, vee)

        # Dynamics
        x_k = ca.vertcat(q, q_dot)
        Ad = ca.DM(self._Ad)
        Bd = ca.DM(self._Bd)
        disc_dyn_expr = Ad @ x_k + Bd @ self.input 

        model = AcadosModel()

        model.disc_dyn_expr = disc_dyn_expr
        model.x = self.state
        model.u = self.input 

        model.x_labels = [r'$q1$ [rad]', r'$\dot{q1}$ [rad/s]', r'$q2$ [rad]', r'$\dot{q2}$ [rad/s]', r'$q3$ [rad]', r'$\dot{q3}$ [rad/s]',
                          r'$q4$ [rad]', r'$\dot{q4}$ [rad/s]', r'$q5$ [rad]', r'$\dot{q5}$ [rad/s]', r'$q6$ [rad]', r'$\dot{q6}$ [rad/s]']
        model.u_labels = [r'$\dot{q1_ref}$ [rad/s]', r'$\dot{q2_ref}$ [rad/s]', r'$\dot{q3_ref}$ [rad/s]', r'$\dot{q4_ref}$ [rad/s]',
                          r'$\dot{q5_ref}$ [rad/s]', r'$\dot{q6_ref}$ [rad/s]']
        model.t_label = '$t$ [s]'

        return model, y
    
    def forward_kinematics_euler(self, q):
        return self.fk_casadi_euler(q)

    def forward_kinematics_quat(self, q): 
        return self.fk_casadi_quat(q)

    def forward_kinematics_rot(self, q): 
        return self.fk_casadi_rot(q)
    
    def differential_kinematics(self, q, q_dot):
        return self.dk_casadi(q, q_dot)
    
