import numpy as np
import pinocchio as pin
from pathlib import Path


# This module implements the state space robot model, taking as reference the

class SixDofRobot:

    # ROOT = Path(__file__).parent
    # URDF_DIR = ROOT / "ur_description/urdf"
    
    # def __init__(self, Ts, T, Tu, z0, u_step, urdf_loader, urdf_file_name):
    def __init__(self, Ts, T, Tu, z0, u_step, urdf_loader):
        self._Ts = Ts
        self._T = T
        self._Tu = Tu
        self._z0 = z0
        self._u_step = u_step

        self._model = urdf_loader.model
        self._data = urdf_loader.data

        # self._urdf_file_path = Robot.URDF_DIR / urdf_file_name
        # self._mesh_dir = [str(self._urdf_file_path.parent)]
        # self._model, vmodel, cmodel = pin.buildModelsFromUrdf(self._urdf_file_path, package_dirs=self._mesh_dir)
        # self._data = self._model.createData()




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
    def T(self):
        return self._T
    
    @T.setter
    def T(self, new_T):
        if new_T <= 0:
            raise ValueError("The final simulation time must be positive")
        if not isinstance(new_T, float):
            raise TypeError(f"Expected type float, got {type(new_T).__name__}.")
        self._T = new_T

    @property
    def Tu(self):
        return self._Tu
    
    @T.setter
    def Tu(self, new_Tu):
        if new_Tu <= 0:
            raise ValueError("The input step time must be positive")
        if not isinstance(new_T, float):
            raise TypeError(f"Expected type float, got {type(new_Tu).__name__}.")
        self._Tu = new_Tu


    @property
    def z0(self):
        return self._z0

    @z0.setter
    def z0(self, new_z0):
        if not (isinstance(new_z0, np.ndarray) and new_z0.shape == (6, 1) and new_z0.dtype == float):
            raise TypeError("Expected a NumPy column vector of shape (6,1) with dtype=float.")
        self._z0 = new_z0

    @property
    def u_step(self):
        return self._u_step

    def _continous_time_model(self, z, u, th):
        # th = (w_cv1, ..., w_cv6)
        Wcp = np.diag(th)
        z_dot = np.zeros((12,), dtype=np.float64)
        y = np.zeros((13,), dtype=np.float64)

        z_dot[:6] = z[:6]                   # Joint positions state equations
        z_dot[6:] = - Wcp@z[6:] + Wcp@u     # Joint speeds state equations 
        y[:7] = self._forward_kinematics(z[:6])  # End effector pose output equations
        y[7:] = self._diff_kinematic(z[:6],z[6:])          # End effector velocities output equations

        return z, y

    def _forward_kinematics(self, q):
        fid = self._model.getFrameId('ee_fixed_joint')

        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        

        R = self._data.oMf[fid].rotation
        p = self._data.oMf[fid].translation

        quat = pin.Quaternion(R).coeffs()
        
        return np.hstack((p, quat)) # [x_e, y_e, z_e, x, y, z, w]


    def _diff_kinematic(self, q, qdot):
        fid = self._model.getFrameId('ee_fixed_joint')
        J = pin.computeFrameJacobian(self._model, self._data, q, fid, pin.ReferenceFrame.WORLD)

        return J@qdot

    def ffd_simulation(self, th):
        N = int(self._T/self._Ts)
        z = np.zeros((12,N), dtype=np.float64)
        u = np.zeros((6,N), dtype=np.float64)
        y = np.zeros((13,N), dtype=np.float64)

        u[:,0] = self._u_step
        z[:,0] = self._z0

        for k in range (0,N-1):
            z_dot, y_ = self._continous_time_model(z[:,k],u[:,k], th)
            z[:,k+1] = z[:,k] + self._Ts*z_dot
            u[:,k+1] = u[:,k]
            y[:,k] = y_

        return z, u
        

    def rk2_simulation(self):
        N = int(self._T/self._Ts)
        z = np.zeros((13,N), dtype=np.float64)
        u = np.zeros((6,N), dtype=np.float64)
        y = np.zeros((13,N), dtype=np.float64)

        u[:,0] = self._u_step
        z[:,0] = self._z0

        for k in range (0,N-1):
            z_dot, y = self._continous_time_model(z[:,k],u[:,k], th)
            z_temp = z[:,k] + 0.5*self._Ts*z_dot
            z[:,k+1] = z[:,k] + self._Ts*self._continous_time_model(z_temp, u[:,k], th)[1]
            u[:,k+1] = u[:,k]
            y[:,k] = y

        return z, u


# w_cv = np.array([100,100,100,100,100,100], dtype=np.float64)


# robot = Robot(0.01,5.0,3.0,z_0,1.0,'ur5_robot.urdf')
# z, u = robot.ffd_simulation(w_cv)








    


