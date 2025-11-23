import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

class SixDofRobot:
    def __init__(self, urdf_loader, integration_method="RK2", dt=0.01, th=None):
        # Numeric Pinocchio model (for loading URDF and reference)
        self._model = urdf_loader.model
        self._data = urdf_loader.data
        
        # CasADi Pinocchio model (for symbolic computations)
        self._cmodel = cpin.Model(self._model)
        self._cdata = self._cmodel.createData()
        
        self._integration_method_name = integration_method
        self.dt = dt  # Fixed timestep for integration
        
        # Get the number of degrees of freedom
        self.n_dof = self._model.nq
        self._z0 = np.zeros(2 * self.n_dof)
        
        # Get end-effector frame ID from both models
        # Use 'tool0' as the end-effector frame (standard UR5 frame)
        self._ee_frame_id = self._model.getFrameId('tool0')
        self._cee_frame_id = self._cmodel.getFrameId('tool0')
        
        # Check if frame ID is valid
        if self._ee_frame_id >= self._model.nframes:
            raise ValueError(f"Frame 'tool0' not found in model. Available frames: {[self._model.frames[i].name for i in range(self._model.nframes)]}")
        
        # Parameters for velocity control (diagonal damping matrix)
        if th is None:
            th = np.ones(self.n_dof)  # Default parameters
        self._th = th
        
        # Define CasADi symbolic variables
        self.state = ca.SX.sym('x', 2 * self.n_dof)  # [q, q_dot]
        self.control = ca.SX.sym('u', self.n_dof)    # control input (velocity commands)
        # Store damping coefficients as numeric constants (not symbolic)
        self._th_numeric = self._th
        
        # Create CasADi functions for Pinocchio operations
        self._setup_casadi_functions()
        
        # Define the dynamics as an ODE
        self.ode = self._generate_dynamics_model()
        
        # Create integrator based on method
        if integration_method == "RK2":
            self._integrator = self._create_integrator("rk", number_of_finite_elements=2)
        elif integration_method == "RK4":
            self._integrator = self._create_integrator("rk", number_of_finite_elements=4)
        elif integration_method == "Euler":
            self._integrator = self._create_integrator("rk", number_of_finite_elements=1)
        elif integration_method == "cvodes":
            self._integrator = self._create_integrator("cvodes")
        elif integration_method == "idas":
            self._integrator = self._create_integrator("idas")
        elif integration_method == "collocation":
            self._integrator = self._create_integrator("collocation")
        else:
            raise ValueError(f"Unknown integration method: {integration_method}")
    
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
        self.fk_casadi = ca.Function('forward_kinematics',
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
        
        # Differential kinematics using CasADi Pinocchio
        J = cpin.computeFrameJacobian(self._cmodel, self._cdata, q_sym,
                                       self._cee_frame_id, pin.ReferenceFrame.WORLD)
        ee_velocity = J @ q_dot_sym
        
        self.dk_casadi = ca.Function('differential_kinematics',
                                      [q_sym, q_dot_sym],
                                      [ee_velocity],
                                      ['q', 'q_dot'], ['ee_vel'])
    
    def _rotation_matrix_to_quaternion_casadi(self, R):
        """
        Convert 3x3 rotation matrix to quaternion [x, y, z, w] using CasADi
        Uses numerically stable implementation based on trace
        
        Args:
            R: 3x3 CasADi rotation matrix
        
        Returns:
            quaternion [x, y, z, w] as CasADi expression
        """
        # Trace of rotation matrix
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        # Standard formula when trace > 0 (most common case)
        # w = 0.5 * sqrt(1 + trace)
        # For numerical stability, add small epsilon
        w = 0.5 * ca.sqrt(1 + trace + 1e-10)
        factor = 0.25 / w
        
        x = (R[2, 1] - R[1, 2]) * factor
        y = (R[0, 2] - R[2, 0]) * factor
        z = (R[1, 0] - R[0, 1]) * factor
        
        return ca.vertcat(x, y, z, w)
    
    def _rotation_matrix_to_euler_casadi(self, R):
        """
        Convert 3x3 rotation matrix to euler angles (ZYX convention) using CasADi
        
        Args:
            R: 3x3 CasADi rotation matrix
        
        Returns:
            euler angles [roll, pitch, yaw] as CasADi expression
        """
        # ZYX Euler angles (roll-pitch-yaw)
        # pitch = atan2(-R[2,0], sqrt(R[0,0]^2 + R[1,0]^2))
        # roll = atan2(R[2,1], R[2,2])
        # yaw = atan2(R[1,0], R[0,0])
        
        pitch = ca.atan2(-R[2, 0], ca.sqrt(R[0, 0]**2 + R[1, 0]**2))
        roll = ca.atan2(R[2, 1], R[2, 2])
        yaw = ca.atan2(R[1, 0], R[0, 0])
        
        return ca.vertcat(roll, pitch, yaw)
    
    def _generate_dynamics_model(self):
        """
        Create the ODE representation: dx/dt = f(x, u)
        
        Dynamics model:
        q_dot = dq/dt (velocity is derivative of position)
        dq_dot/dt = -diag(th) @ q_dot + diag(th) @ u
        
        This is a first-order velocity control model with damping.
        The damping coefficients (th) are constant numeric values.
        """
        # Split state into position and velocity
        q = self.state[:self.n_dof]
        q_dot = self.state[self.n_dof:]
        
        # Diagonal damping matrix from numeric constant parameters
        Wcp = ca.diag(self._th_numeric)
        
        # State derivatives
        q_dot_derivative = q_dot  # dq/dt = q_dot
        q_ddot = -Wcp @ q_dot + Wcp @ self.control  # dq_dot/dt
        
        # State derivative: [q_dot, q_ddot]
        state_dot = ca.vertcat(q_dot_derivative, q_ddot)
        
        # Define ODE structure (only state and control as inputs)
        ode = {
            'x': self.state,      # state
            'p': self.control,    # parameters (control input only)
            'ode': state_dot      # dx/dt
        }

        return ode
    
    def _create_integrator(self, integrator_type, **options):
        """Create CasADi integrator with specified method and fixed timestep"""

        integrator = ca.integrator(
            f'integrator_{self._integration_method_name}',
            integrator_type,
            self.ode,
            0,      # t0
            self.dt,  # tf (fixed timestep)
            options
        )
        
        return integrator
    
    def update(self, state, control=None):
        """
        Integrate the system forward by one timestep using CasADi integrator
        
        Args:
            state: Current state [q, q_dot] as numpy array
            control: Control input (velocity commands). If None, uses zeros.
        
        Returns:
            new_state: Integrated state as numpy array
        """
        if control is None:
            control = np.zeros(self.n_dof)
        
        # Call integrator with control as parameter (timestep is fixed at initialization)
        result = self._integrator(x0=state, p=control)
        
        # Extract and return new state
        new_state = np.array(result['xf']).flatten()
        return new_state
    
    def forward_kinematics(self, q):
        """
        Compute forward kinematics for given joint positions (Euler angles version)
        
        Args:
            q: Joint positions (CasADi symbolic or numeric)
        
        Returns:
            End-effector pose [position(3), euler_angles(3)] as CasADi expression
        """
        return self.fk_casadi_euler(q)
    
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
    
    def get_explicit_model(self):
        """Return explicit ODE model with outputs"""
        q = self.state[:self.n_dof]
        q_dot = self.state[self.n_dof:]
        
        # Outputs: [end_effector_pose(12 with rotmat or 6 with euler), end_effector_velocity(6)]
        ee_pose_rotmat = self.fk_casadi(q)  # position + rotation matrix
        ee_vel = self.dk_casadi(q, q_dot)
        output = ca.vertcat(ee_pose_rotmat, ee_vel)
        
        return {
            'state': self.state,
            'control': self.control,
            'ode': self.ode['ode'],
            'output': output
        }
    
    def get_implicit_model(self):
        """Return implicit (DAE) model"""
        # For explicit ODE: 0 = dx/dt - f(x, u)
        residual = self.state - self.ode['ode']
        return {
            'state': self.state,
            'control': self.control,
            'residual': residual
        }
    
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