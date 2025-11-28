import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin
from acados_template import AcadosModel


class SixDofRobot:
    def __init__(self, urdf_loader, Wcv=None):
        # Numeric Pinocchio model (for loading URDF and reference)
        self._model = urdf_loader.model
        self._data = urdf_loader.data
        
        # CasADi Pinocchio model (for symbolic computations)
        self._cmodel = cpin.Model(self._model)
        self._cdata = self._cmodel.createData()
        
        # Get the number of degrees of freedom
        self.n_dof = self._model.nq
        
        # Get end-effector frame ID from both models
        # Use 'tool0' as the end-effector frame (standard UR5 frame)
        self._ee_frame_id = self._model.getFrameId('tool0')
        self._cee_frame_id = self._cmodel.getFrameId('tool0')
        
        # Check if frame ID is valid
        if self._ee_frame_id >= self._model.nframes:
            raise ValueError(f"Frame 'tool0' not found in model. Available frames: {[self._model.frames[i].name for i in range(self._model.nframes)]}")
        
        # Speed loop bandwidths (diagonal matrix)
        if Wcv is None:
            Wcv = np.ones(self.n_dof)  # Default parameters
        self._Wcv = Wcv
        
        # Define CasADi symbolic variables
        self.state = ca.SX.sym('x', 2 * self.n_dof)  # [q, q_dot]
        self.input = ca.SX.sym('u', self.n_dof)    # control input (velocity commands)
        
        # Create CasADi functions for Pinocchio operations
        self._setup_casadi_functions()
        
        # Define the dynamics as an ODE
        self.acados_model, self.y = self._generate_dynamics_model()

    def _setup_casadi_functions(self):
        """Setup pure CasADi functions for Pinocchio operations"""
        # Create symbolic variables for Pinocchio functions
        q_sym = ca.SX.sym('q', self.n_dof)
        q_dot_sym = ca.SX.sym('q_dot', self.n_dof)
        
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
        
        
        # Convert rotation matrix to euler angles (ZYX convention)
        euler_angles = self._rotation_matrix_to_euler_casadi(ee_rotation)

        # Concatenate position and euler angles [x, y, z, roll, pitch, yaw]
        ee_pose_euler = ca.vertcat(ee_position, euler_angles)
        
        # Create FK function with euler angles output
        self.fk_casadi_euler = ca.Function('forward_kinematics_euler',
                                            [q_sym],
                                            [ee_pose_euler],
                                            ['q'], ['pose'])
        
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
        
        return ca.vertcat(roll, pitch, yaw)
    

    def _generate_dynamics_model(self) -> AcadosModel:
        """
        Create the ODE representation: dx/dt = f(x, u)
        
        Dynamics model:
        dq/dt = q_dot (integrator for the position)
        dq_dot/dt = -diag(th) @ q_dot + diag(th) @ u
        
        This is a first-order, decoupled, speed closed-loop model of the robot.
        """
        # States
        q = self.state[:self.n_dof]
        q_dot = self.state[self.n_dof:]

        # Output 
        pose_ee_rot  = self.fk_casadi_rot(q)  # position + rotation matrix
        vee = self.dk_casadi(q, q_dot)
        y = ca.vertcat(pose_ee_rot, vee)

        # Constants
        Wcv = ca.diag(self._Wcv)
        
        # State derivatives
        xdot = ca.SX.sym('xdot', 2*self.n_dof)

        # Dynamics
        f_expl = ca.vertcat(q_dot, -Wcv @ q_dot + Wcv @ self.input)
        f_impl = xdot - f_expl

        model = AcadosModel()

        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = self.state
        model.xdot = xdot
        model.u = self.input 

        model.x_labels = [r'$q1$ [rad]', r'$\dot{q1}$ [rad/s]', r'$q2$ [rad]', r'$\dot{q2}$ [rad/s]', r'$q3$ [rad]', r'$\dot{q3}$ [rad/s]',
                          r'$q4$ [rad]', r'$\dot{q4}$ [rad/s]', r'$q5$ [rad]', r'$\dot{q5}$ [rad/s]', r'$q6$ [rad]', r'$\dot{q6}$ [rad/s]']
        model.u_labels = [r'$\dot{q1_ref}$ [rad/s]', r'$\dot{q2_ref}$ [rad/s]', r'$\dot{q3_ref}$ [rad/s]', r'$\dot{q4_ref}$ [rad/s]',
                          r'$\dot{q5_ref}$ [rad/s]', r'$\dot{q6_ref}$ [rad/s]']
        model.t_label = '$t$ [s]'

        return model, y
    
    def forward_kinematics_euler(self, q):
        """
        Compute forward kinematics for given joint positions (Euler angles version)
        
        Args:
            q: Joint positions (CasADi symbolic or numeric)
        
        Returns:
            End-effector pose [position(3), euler_angles(3)] as CasADi expression
        """
        return self.fk_casadi_euler(q)

    def forward_kinematics_quat(self, q): 
        """FK: position + quaternion"""
        return self.fk_casadi_quat(q)

    def forward_kinematics_rot(self, q): 
        """FK: position + quaternion"""
        return self.fk_casadi_rot(q)
    
    def differential_kinematics(self, q, q_dot):
        """
        Compute differential kinematics (end-effector velocity)
        
        Args:
            q: Joint positions (CasADi symbolic or numeric)
            q_dot: Joint velocities (CasADi symbolic or numeric)
        
        Returns:
            End-effector velocity [linear_vel(3), angular_vel(3)] as CasADi expression
        """
        return self.dk_casadi(q, q_dot)
    

    def set_initial_state(self, initial_state):
        self._z0 = initial_state
    
    @property
    def state_dimensions(self):
        return 2 * self.n_dof
    
    @property
    def control_dimensions(self):
        return self.n_dof
    
    @property
    def initial_state(self):
        return self._z0