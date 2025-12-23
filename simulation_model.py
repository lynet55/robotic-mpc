import numpy as np
import pinocchio as pin
from loader import UrdfLoader
from numpy.typing import NDArray
FloatArray = NDArray[np.float64]

  

# This module implements the state space robot model, taking as reference the
# ur5 6 d.o.f industrial manipulator, using different simulation approaches.
class Robot:

    def __init__(self, z0, u0, Nsim, dt, wcv, urdf_loader: UrdfLoader, integration_method="RK4") -> None:
        self.urdf_loader = urdf_loader
        self._model = urdf_loader.model
        self._data = urdf_loader.data
        self._fee = urdf_loader.fee

        self._dt = dt
        self._z0 = z0
        self._u0 = u0
        self._N = Nsim
        self.wcv = np.diag(wcv)

        self.z = np.zeros((12,self._N + 1), dtype=np.float64)
        self.u = np.zeros((6,self._N + 1), dtype=np.float64)
        self._ee_rpy_log = np.zeros((3,self._N + 1), dtype=np.float64)
        self._ee_pose_log = np.zeros((12,self._N + 1), dtype=np.float64)
        self._ee_velocity_log = np.zeros((6,self._N + 1), dtype=np.float64)

        self._t = 0
        self.z[:, 0] = self._z0
        self.u[:, 0] = self._u0
        self._ee_pose_log[:, 0], self._ee_rpy_log[:, 0] = self._forward_kinematics(self._z0[:6])
        self._ee_velocity_log[:, 0] = self._diff_kinematic(self._z0[:6], self._z0[6:])

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
    def dt(self):
        return self._dt
    @property
    def N(self):
        return self._N

    def _forward_kinematics(self, q):
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        R = self._data.oMf[self._fee].rotation
        p = self._data.oMf[self._fee].translation
        R_flat = R.reshape(9,)
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
        yaw = np.arctan2(R[1, 0], R[0, 0])  
        pose = np.hstack((p, R_flat))
        rpy  = np.array([roll, pitch, yaw])
        return pose, rpy
    
    
    def _diff_kinematic(self, q, qdot):
        # The WORLD frame is retrieved by Pinocchio from the urdf, where by default is set as coincident with the robot base frame
        J = pin.computeFrameJacobian(self._model, self._data, q, self._fee, pin.ReferenceFrame.WORLD)
        return J @ qdot  # [vx, vy, vz, wx, wy, wz]

    def _continuos_time_state(self, z, u):
        z_dot = np.zeros((12,), dtype=np.float64)
        z_dot[:6] = z[6:]  # Joint positions state equations
        z_dot[6:] = -self.wcv @ z[6:] + self.wcv @ u  # Joint speeds state equations
        return z_dot

    def update(self, z_k, u_k, t):
        next_state = self._integration_method(z_k, u_k, self._dt)
        self.z[:, t + 1] = next_state
        self.u[:, t + 1] = u_k
        self._ee_pose_log[:, t + 1], self._ee_rpy_log[:, t + 1] = self._forward_kinematics(next_state[:6])
        self._ee_velocity_log[:, t + 1] = self._diff_kinematic(next_state[:6], next_state[6:])
        return next_state

    def _euler_update(self, z_k, u_k, dt):
        """Forward Euler integration (RK1)."""
        k1 = self._continuos_time_state(z_k, u_k)
        return z_k + dt * k1

    def _rk2_update(self, z_k, u_k, dt):
        """Runge-Kutta 2nd order (midpoint method)."""
        k1 = self._continuos_time_state(z_k, u_k)
        k2 = self._continuos_time_state(z_k + 0.5 * dt * k1, u_k)
        return z_k + self.dt * k2

    def _rk3_update(self, z_k, u_k, dt):
        """Runge-Kutta 3rd order."""
        k1 = self._continuos_time_state(z_k, u_k)
        k2 = self._continuos_time_state(z_k + 0.5 * dt * k1, u_k)
        k3 = self._continuos_time_state(z_k - dt * k1 + 2 * dt * k2, u_k)
        return z_k + (dt / 6) * (k1 + 4 * k2 + k3)

    def _rk4_update(self, z_k, u_k, dt):
        """Runge-Kutta 4th order."""
        k1 = self._continuos_time_state(z_k, u_k)
        k2 = self._continuos_time_state(z_k + 0.5 * dt * k1, u_k)
        k3 = self._continuos_time_state(z_k + 0.5 * dt * k2, u_k)
        k4 = self._continuos_time_state(z_k + dt * k3, u_k)
        return z_k + (dt / 6) * k1 + (dt / 3) * k2 + (dt / 3) * k3 + (dt / 6) * k4
    
    def simulate_output(self):
        pose = np.zeros((12,self._N+1), dtype=np.float64)
        v_ee = np.zeros((6,self._N+1), dtype=np.float64)

        for k in range (0,self._N+1):
            q = self.z[:6,k]; qdot = self.z[6:,k]
            pose[:,k], _ = self._forward_kinematics(q)
            v_ee[:,k] = self._diff_kinematic(q, qdot)

        return pose, v_ee

    def get_state(self) -> FloatArray:
        return self.z[:, -1]

    def get_control_input(self) -> FloatArray:
        return self.u[:, -1]

    def ee_position(self, t: int) -> FloatArray:
        """Return end-effector position at the specified t."""
        return self._ee_pose_log[:3, t]

    def ee_orientation_euler(self, t: int) -> FloatArray:
        """Return end-effector orientation as RPY at the specified t."""
        return self._ee_rpy_log[:, t]
    
    def ee_orientation_rot(self, t: int) -> FloatArray:
        """Return end-effector orientation as R flatten at the specified t."""
        return self._ee_pose_log[3:12, t]

    def ee_velocity(self, t: int) -> FloatArray:
        """Return end-effector velocity at the specified t."""
        return self._ee_velocity_log[:, t]

    def joint_angles(self, t: int) -> FloatArray:
        """Return joint positions at the specified t."""
        return self.z[:6, t]

    def joint_velocities(self, t: int) -> FloatArray:
        """Return joint velocities at the specified t."""
        return self.z[6:, t]

    def state(self, t: int) -> FloatArray:
        """Return a joint state vector from the simulation history at the specified t."""
        return self.z[:, t]

    def ee_pose(self, t: int) -> FloatArray:
        """Return end-effector pose at the specified t."""
        return self._ee_pose_log[:, t]