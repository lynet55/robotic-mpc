import numpy as np
import pinocchio as pin
import time
from loader import UrdfLoader
from visualization import Visualization
from scipy.linalg import expm
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# This module implements the state space robot model, taking as reference the
# ur5 6 d.o.f industrial manipulator, using different simulation approaches.
class Robot:

    def __init__(self, z0, u0, T, Ts, wcv, urdf_loader: UrdfLoader, integration_method="RK4") -> None:
        self.urdf_loader = urdf_loader
        self._model = urdf_loader.model
        self._data = urdf_loader.data
        self._fee = urdf_loader.fee

        self._Ts = Ts
        self._T = T
        self._z0 = z0
        self._u0 = u0
        self._N = int(T/Ts)
        self.wcv = np.diag(wcv)

        self.z = np.zeros((12,self._N), dtype=np.float64)
        self.u = np.zeros((6,self._N), dtype=np.float64)
        self._ee_pose_log = np.zeros((6,self._N), dtype=np.float64)
        self._ee_velocity_log = np.zeros((6,self._N), dtype=np.float64)

        self._current_index = 0
        self.z[:, 0] = self._z0
        self.u[:, 0] = self._u0
        self._ee_pose_log[:, 0] = self.forward_kinematics(self._z0[:6])
        self._ee_velocity_log[:, 0] = self.diff_kinematic(self._z0[:6], self._z0[6:])
        
        self._integration_method_name = integration_method
        if integration_method == "Euler":
            self._integration_method = self._euler_update
        elif integration_method == "RK2":
            self._integration_method = self._rk2_update
        elif integration_method == "RK3":
            self._integration_method = self._rk3_update
        elif integration_method == "RK4":
            self._integration_method = self._rk4_update
        else:
            raise ValueError(f"Unknown integration method: {integration_method}")
   
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
        self._update_N_and_Nu()

    @property
    def T(self):
        return self._T
    @property
    def N(self):
        return self._N
    
    @T.setter
    def T(self, new_T):
        if new_T <= 0:
            raise ValueError("The total simulation time must be positive")
        if not isinstance(new_T, float):
            raise TypeError(f"Expected type float, got {type(new_T).__name__}.")
        self._T = new_T
        self._update_N_and_Nu()

    @property
    def Tu(self):
        return self._Tu
    
    @Tu.setter
    def Tu(self, new_Tu):
        if new_Tu <= 0:
            raise ValueError("The input step time must be positive")
        if not isinstance(new_Tu, float):
            raise TypeError(f"Expected type float, got {type(new_Tu).__name__}.")
        self._Tu = new_Tu
        self._update_N_and_Nu()


    @property
    def z0(self):
        return self._z0

    # @z0.setter
    # def z0(self, new_z0):
    #     if not (isinstance(new_z0, np.ndarray) and new_z0.shape == (12,) and new_z0.dtype == float):
    #         raise TypeError("Expected a NumPy column vector of shape (12,1) with dtype=float.")
    #     self._z0 = new_z0

    # @property
    # def u_step(self):
    #     return self._u_step

    # @u_step.setter
    # def u_step(self, new_u):
    #     if not isinstance(new_u, float):
    #         raise TypeError(f"Expected type float, got {type(new_u).__name__}.")
    #     self._u_step = new_u

    # def _update_N_and_Nu(self):
    #     self._N = int(self._T / self._Ts)
    #     self._Nu = int(self._Tu / self._Ts)

    def forward_kinematics(self, q):
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        
        R = self._data.oMf[self._fee].rotation
        p = self._data.oMf[self._fee].translation
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
        yaw = np.arctan2(R[1, 0], R[0, 0])

        # quat = pin.Quaternion(R).coeffs()
        
        return np.hstack((p, roll, pitch, yaw))                # [x_e, y_e, z_e, roll, pitch, yaw]

    def diff_kinematic(self, q, qdot):
        # The WORLD frame is retrieved by Pinocchio from the urdf, where by default is set as coincident with the robot base frame
        J = pin.computeFrameJacobian(self._model, self._data, q, self._fee, pin.ReferenceFrame.WORLD)
        return J@qdot                              # [vx, vy, vz, wx, wy, wz]

    def _continuos_time_state(self, z, u, Wcv):
        z_dot = np.zeros((12,), dtype=np.float64)
        z_dot[:6] = z[6:]                          # Joint positions state equations
        z_dot[6:] = - self.wcv @ z[6:] + self.wcv @ u        # Joint speeds state equations
        return z_dot

    def _euler_update(self, z_k, u_k, dt):
        """Forward Euler integration (RK1)."""
        k1 = self._continuos_time_state(z_k, u_k, self.wcv)
        return z_k + dt * k1

    def _rk2_update(self, z_k, u_k, dt):
        """Runge-Kutta 2nd order (midpoint method)."""
        k1 = self._continuos_time_state(z_k, u_k, self.wcv)
        k2 = self._continuos_time_state(z_k + 0.5 * dt * k1, u_k, self.wcv)
        return z_k + dt * k2

    def _rk3_update(self, z_k, u_k, dt):
        """Runge-Kutta 3rd order."""
        k1 = self._continuos_time_state(z_k, u_k, self.wcv)
        k2 = self._continuos_time_state(z_k + 0.5 * dt * k1, u_k, self.wcv)
        k3 = self._continuos_time_state(z_k - dt * k1 + 2 * dt * k2, u_k, self.wcv)
        return z_k + (dt / 6) * (k1 + 4 * k2 + k3)

    def _rk4_update(self, z_k, u_k, dt):
        """Runge-Kutta 4th order."""
        k1 = self._continuos_time_state(z_k, u_k, self.wcv)
        k2 = self._continuos_time_state(z_k + 0.5 * dt * k1, u_k, self.wcv)
        k3 = self._continuos_time_state(z_k + 0.5 * dt * k2, u_k, self.wcv)
        k4 = self._continuos_time_state(z_k + dt * k3, u_k, self.wcv)
        return z_k + (dt / 6) * k1 + (dt / 3) * k2 + (dt / 3) * k3 + (dt / 6) * k4

    def update(self, z_k, u_k, t):
        next_state = self._integration_method(z_k, u_k, self._Ts)
        self.z[:, t + 1] = next_state
        self.u[:, t + 1] = u_k
        self._ee_pose_log[:, t + 1] = self.forward_kinematics(next_state[:6])
        self._ee_velocity_log[:, t + 1] = self.diff_kinematic(next_state[:6], next_state[6:])
        return next_state
    
    def euler_simulation(self):
        start = time.perf_counter()
        z = np.zeros((12,self._N), dtype=np.float64)
        u = np.zeros((6,self._N), dtype=np.float64)

        u[:,0] = self._u0
        z[:,0] = self._z0

        for k in range (0,self._N-1):
            # k1 = self._continuos_time_state(z[:,k],u[:,k], Wcv) #z_dot, output = ...
            z[:,k+1] = self._euler_update(z[:,k], u[:,k], self._Ts)
            u[:, k+1] = u[:, k] if (k + 1) < self._Nu else np.zeros((6,), dtype=np.float64)

        end = time.perf_counter()
        print(f"Elapsed time for forward finite difference simulation: {(end - start) * 1e3:.3f} ms")

        return z, u
        

    def rk2_mp_simulation(self):
        start = time.perf_counter()
        z = np.zeros((12,self._N), dtype=np.float64)
        u = np.zeros((6,self._N), dtype=np.float64)

        u[:,0] = self._u0
        z[:,0] = self._z0

        for k in range (0,self._N-1):
            # k1 = self._continuos_time_state(z[:,k],u[:,k], Wcv)
            # k2 = self._continuos_time_state(z[:,k] + 0.5*self._Ts*k1,u[:,k], Wcv)
            z[:,k+1] = self._rk2_update(z[:,k], u[:,k], self._Ts)
            u[:, k+1] = u[:, k] if (k + 1) < self._Nu else np.zeros((6,), dtype=np.float64)
        
        end = time.perf_counter()
        print(f"Elapsed time for rk2 middle-point simulation: {(end - start) * 1e3:.3f} ms")
        
        return z, u

    def rk4_simulation(self):
        start = time.perf_counter()

        z = np.zeros((12,self._N), dtype=np.float64)
        u = np.zeros((6,self._N), dtype=np.float64)

        u[:,0] = self._u0
        z[:,0] = self._z0

        for k in range (0,self._N-1):
            z[:,k+1] = self._rk4_update(z[:,k], u[:,k], self._Ts)
            u[:, k+1] = u[:, k] if (k + 1) < self._Nu else np.zeros((6,), dtype=np.float64)
        
        end = time.perf_counter()
        print(f"Elapsed time for rk4 simulation: {(end - start) * 1e3:.3f} ms")
        
        return z, u
    

    def lagrange_formula(self, Tu): 
        start = time.perf_counter() 

        z = np.zeros((12, self._N), dtype=np.float64) 
        u = np.zeros((6, self._N), dtype=np.float64)
        Nu = int(Tu/self._Ts)

        for k in range(self._N):
            u[:, k] = self._u0 if k < Nu else 0.0

        Ts = self._Ts
        Tu = Nu * Ts

        for i in range(6):
            w = wcv[i]
            u_step = self._u_step

            # Values at the end of the step
            v_Tu = u_step * (1.0 - np.exp(-w * Tu))
            q_Tu = u_step * Tu + (u_step / w) * (np.exp(-w * Tu) - 1.0)

            for k in range(self._N):
                t = k * Ts

                if k < self._Nu:
                    # First phase: costant input = u_step
                    v = u_step * (1.0 - np.exp(-w * t))
                    q = u_step * t + (u_step / w) * (np.exp(-w * t) - 1.0)
                else:
                    # Second phase: null input, yet with non zero initial condition
                    dt = t - Tu
                    v = v_Tu * np.exp(-w * dt)
                    q = q_Tu + (v_Tu / w) * (1.0 - np.exp(-w * dt))

                z[i,   k] = q        
                z[i+6, k] = v        
                    
        end = time.perf_counter() 
        print(f"Elapsed time for exact simulation: {(end - start) * 1e3:.3f} ms") 
        return z, u


    def compute_output(self, z):
        y = np.zeros((13,self._N), dtype=np.float64)

        for k in range (0,self._N):
            q = z[:6,k]; qdot = z[6:,k]
            y[:7,k] = self._forward_kinematics(q)
            y[7:,k] = self._diff_kinematic(q, qdot)

        return y
    
    def compute_inertia_matrix(self, q):
        model = self._urdf_loader.model
        data = self._urdf_loader.data
    
        # Calcolo della matrice di inerzia tramite CRBA
        M = pin.crba(model, data, q)
        M = (M + M.T) * 0.5       # Simmetrizzazione numerica, consigliata da Pinocchio
        return M
    
    def get_state(self, index: int) -> FloatArray:
        return self.z[:, -1]
    
    def get_control_input(self, index: int) -> FloatArray:
        return self.u[:,-1]

    def ee_position(self, index: int) -> FloatArray:
        """Return end-effector position at the specified index."""
        return self._ee_pose_log[:3, index]
    
    def ee_orientation(self, index: int) -> FloatArray:
        """Return end-effector orientation at the specified index."""
        return self._ee_pose_log[3:6, index]

    def ee_velocity(self, index: int) -> FloatArray:
        """Return end-effector velocity at the specified index."""
        return self._ee_velocity_log[:, index]
    
    def joint_angles(self, index: int) -> FloatArray:
        """Return joint positions at the specified index."""
        return self.z[:6, index]

    def joint_velocities(self, index: int) -> FloatArray:
        """Return joint velocities at the specified index."""
        return self.z[6:, index]

    def state(self, index: int) -> FloatArray:
        """Return a joint state vector from the simulation history at the specified index."""
        return self.z[:, index]

    def ee_pose(self, index: int) -> FloatArray:
        """Return end-effector pose at the specified index."""
        return self._ee_pose_log[:, index]
    
    @property
    def ee_pose_log(self) -> FloatArray:
        """Return the full end-effector pose log."""
        return self._ee_pose_log
    
    @property
    def ee_velocity_log(self) -> FloatArray:
        """Return the full end-effector velocity log."""
        return self._ee_velocity_log

    






    



