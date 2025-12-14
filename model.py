import numpy as np
import pinocchio as pin
import time
from loader import UrdfLoader
from scipy.linalg import expm

# This module implements the state space robot model, taking as reference the
# ur5 6 d.o.f industrial manipulator, using different simulation approaches.
class Robot:
    
    def __init__(self, Ts:float, T:float, Tu:float, z0, u_step:float, urdf_loader: UrdfLoader) -> None:
        self._Ts = Ts
        self._T = T
        self._Tu = Tu
        self._z0 = z0
        self._u_step = u_step
        self._urdf_loader = urdf_loader

        self._N = int(self._T/self._Ts)
        self._Nu = int(self._Tu/self._Ts)
   
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

    @z0.setter
    def z0(self, new_z0):
        if not (isinstance(new_z0, np.ndarray) and new_z0.shape == (12,) and new_z0.dtype == float):
            raise TypeError("Expected a NumPy column vector of shape (12,1) with dtype=float.")
        self._z0 = new_z0

    @property
    def u_step(self):
        return self._u_step

    @u_step.setter
    def u_step(self, new_u):
        if not isinstance(new_u, float):
            raise TypeError(f"Expected type float, got {type(new_u).__name__}.")
        self._u_step = new_u

    def _update_N_and_Nu(self):
        self._N = int(self._T / self._Ts)
        self._Nu = int(self._Tu / self._Ts)

    def _forward_kinematics(self, q):
        model = self._urdf_loader.model
        data = self._urdf_loader.data
        fee = self._urdf_loader.fee

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        R = data.oMf[fee].rotation
        p = data.oMf[fee].translation

        quat = pin.Quaternion(R).coeffs()
        
        return np.hstack((p, quat))                # [x_e, y_e, z_e, x, y, z, w]

    def _diff_kinematic(self, q, qdot):
        model = self._urdf_loader.model
        data = self._urdf_loader.data
        fee = self._urdf_loader.fee
        
        # The WORLD frame is retrieved by Pinocchio from the urdf, where by default is set as coincident with the robot base frame
        J = pin.computeFrameJacobian(model, data, q, fee, pin.ReferenceFrame.WORLD)

        return J@qdot                              # [vx, vy, vz, wx, wy, wz]

    def _continuos_time_state(self, z, u, Wcv):
        z_dot = np.zeros((12,), dtype=np.float64)
        z_dot[:6] = z[6:]                          # Joint positions state equations
        z_dot[6:] = - Wcv @ z[6:] + Wcv @ u        # Joint speeds state equations
        return z_dot

    def euler_simulation(self, wcv):
        start = time.perf_counter()
        Wcv = np.diag(wcv)

        z = np.zeros((12,self._N), dtype=np.float64)
        u = np.zeros((6,self._N), dtype=np.float64)

        u[:,0] = self._u_step
        z[:,0] = self._z0

        for k in range (0,self._N-1):
            k1 = self._continuos_time_state(z[:,k],u[:,k], Wcv) #z_dot, output = ...
            z[:,k+1] = z[:,k] + self._Ts*k1

            u[:, k+1] = u[:, k] if (k + 1) < self._Nu else np.zeros((6,), dtype=np.float64)

        end = time.perf_counter()
        print(f"Elapsed time for forward finite difference simulation: {(end - start) * 1e3:.3f} ms")

        return z, u
        

    def rk2_mp_simulation(self, wcv):
        start = time.perf_counter()
        Wcv = np.diag(wcv)

        z = np.zeros((12,self._N), dtype=np.float64)
        u = np.zeros((6,self._N), dtype=np.float64)

        u[:,0] = self._u_step
        z[:,0] = self._z0

        for k in range (0,self._N-1):
            k1 = self._continuos_time_state(z[:,k],u[:,k], Wcv)
            k2 = self._continuos_time_state(z[:,k] + 0.5*self._Ts*k1,u[:,k], Wcv)
            z[:,k+1] = z[:,k] + self._Ts*k2

            u[:, k+1] = u[:, k] if (k + 1) < self._Nu else np.zeros((6,), dtype=np.float64)
        
        end = time.perf_counter()
        print(f"Elapsed time for rk2 middle-point simulation: {(end - start) * 1e3:.3f} ms")
        
        return z, u

    def rk4_simulation(self, wcv):
        start = time.perf_counter()
        Wcv = np.diag(wcv)

        z = np.zeros((12,self._N), dtype=np.float64)
        u = np.zeros((6,self._N), dtype=np.float64)

        u[:,0] = self._u_step
        z[:,0] = self._z0

        for k in range (0,self._N-1):
            k1 = self._continuos_time_state(z[:,k],u[:,k], Wcv)
            k2 = self._continuos_time_state(z[:,k] + 0.5*self._Ts*k1,u[:,k], Wcv)
            k3 = self._continuos_time_state(z[:,k] + 0.5*self._Ts*k2,u[:,k], Wcv)
            k4 = self._continuos_time_state(z[:,k] +     self._Ts*k3,u[:,k], Wcv)
            z[:,k+1] = z[:,k] + self._Ts/6*k1 + self._Ts/3*k2 + self._Ts/3*k3 + self._Ts/6*k4

            u[:, k+1] = u[:, k] if (k + 1) < self._Nu else np.zeros((6,), dtype=np.float64)
        
        end = time.perf_counter()
        print(f"Elapsed time for rk4 simulation: {(end - start) * 1e3:.3f} ms")
        
        return z, u

    def lagrange_formula(self, wcv): 
        start = time.perf_counter() 

        z = np.zeros((12, self._N), dtype=np.float64) 
        u = np.zeros((6, self._N), dtype=np.float64)

        for k in range(self._N):
            u[:, k] = self._u_step if k < self._Nu else 0.0

        Ts = self._Ts
        Tu = self._Nu * Ts

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



        






















    



