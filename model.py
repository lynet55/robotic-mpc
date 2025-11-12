import numpy as np
import pinocchio as pin
from pathlib import Path
from loader import UrdfLoader


# This module implements the state space robot model, taking as reference the
# ur5 6 d.o.f industrial manipulator, using different simulation approaches.

class Robot:
    '''
    ROOT = Path(__file__).parent
    URDF_DIR = ROOT / "ur_description/urdf"
    '''
    
    def __init__(self, Ts:float, T, Tu, z0, u_step, urdf_loader: UrdfLoader) -> None:
        self._Ts = Ts
        self._T = T
        self._Tu = Tu
        self._z0 = z0
        self._u_step = u_step
        self._urdf_loader = urdf_loader
        '''
        self._urdf_file_path = Robot.URDF_DIR / urdf_file_name
        self._mesh_dir = [str(self._urdf_file_path.parent)]

        try:
            self._model, vmodel, cmodel = pin.buildModelsFromUrdf(self._urdf_file_path, package_dirs=self._mesh_dir)
            print(f"URDF successfully loaded: {self._urdf_file_path}")
            print(f"nq = {self._model.nq}, ngeoms(vis) = {vmodel.ngeoms}, ngeoms(col) = {cmodel.ngeoms}")
            self._data = self._model.createData()

        except FileNotFoundError as e:
            print(f"File not found: {e.filename}")

        except ValueError as e:
            # Invalid URDF or missing mesh files
            print(f"Error while parsing the URDF or building the model:\n{e}")

        except Exception as e:
            # Generic catch for unexpected errors (e.g., missing libraries, permission issues)
            print(f"Unexpected error while loading the model:\n{type(e).__name__}: {e}")

        else:
            print("Pinocchio model and data successfully created.")
    '''
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
            raise ValueError("The total simulation time must be positive")
        if not isinstance(new_T, float):
            raise TypeError(f"Expected type float, got {type(new_T).__name__}.")
        self._T = new_T

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

    def _forward_kinematics(self, q):
        model = self._urdf_loader.model
        data = self._urdf_loader.data
        fee = self._urdf_loader.fee

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        R = data.oMf[fee].rotation
        p = data.oMf[fee].translation

        quat = pin.Quaternion(R).coeffs()
        
        return np.hstack((p, quat))                 # [x_e, y_e, z_e, x, y, z, w]

    

    def _diff_kinematic(self, q, qdot):
        model = self._urdf_loader.model
        data = self._urdf_loader.data
        fee = self._urdf_loader.fee
        

        # The WORLD frame is retrieved by Pinocchio from the urdf, where by default is set as coincident with the robot base frame
        J = pin.computeFrameJacobian(model, data, q, fee, pin.ReferenceFrame.WORLD)

        return J@qdot                                # [vx, vy, vz, wx, wy, wz]

    def _continuos_time_model(self, z, u, Wcp):
        z_dot = np.zeros((12,), dtype=np.float64)
        y = np.zeros((13,), dtype=np.float64)

        z_dot[:6] = z[6:]                           # Joint positions state equations
        z_dot[6:] = - Wcp@z[6:] + Wcp@u             # Joint speeds state equations 
        y[:7] = self._forward_kinematics(z[:6])     # End effector pose output equations
        y[7:] = self._diff_kinematic(z[:6],z[6:])   # End effector velocities output equations

        return z_dot, y

    def ffd_simulation(self, wcp):
        Wcp = np.diag(wcp)
        N = int(self._T/self._Ts)
        Nu = int(self._Tu/self._Ts)
        z = np.zeros((12,N), dtype=np.float64)
        u = np.zeros((6,N), dtype=np.float64)
        y = np.zeros((13,N), dtype=np.float64)

        u[:,0] = self._u_step
        z[:,0] = self._z0

        for k in range (0,N-1):
            z_dot, output = self._continuos_time_model(z[:,k],u[:,k], Wcp)
            z[:,k+1] = z[:,k] + self._Ts*z_dot

            y[:,k] = output
            u[:, k+1] = u[:, k] if (k + 1) < Nu else np.zeros((6,), dtype=np.float64)
        
        _, y[:, -1] = self._continuos_time_model(z[:, -1], u[:, -1], Wcp)

        return z, y, u
        

    def rk2_simulation(self, wcp):
        Wcp = np.diag(wcp)
        N = int(self._T/self._Ts)
        Nu = int(self._Tu/self._Ts)
        z = np.zeros((12,N), dtype=np.float64)
        u = np.zeros((6,N), dtype=np.float64)
        y = np.zeros((13,N), dtype=np.float64)

        u[:,0] = self._u_step
        z[:,0] = self._z0

        for k in range (0,N-1):
            z_dot, output = self._continuos_time_model(z[:,k],u[:,k], Wcp)
            z_mid = z[:,k] + 0.5*self._Ts*z_dot

            z_dot_mid, _ = self._continuos_time_model(z_mid,u[:,k], Wcp)
            z[:,k+1] = z[:,k] + self._Ts*z_dot_mid

            y[:,k] = output
            u[:, k+1] = u[:, k] if (k + 1) < Nu else np.zeros((6,), dtype=np.float64)
        
        _, y[:, -1] = self._continuos_time_model(z[:, -1], u[:, -1], Wcp)

        return z, y, u


q_0 = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
qdot_0 = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)

z_0 = np.hstack((q_0, qdot_0))
w_cp = np.array([100,100,100,100,100,100], dtype=np.float64)

urdf = UrdfLoader('ur5_robot')
robot = Robot(0.01,5.0,3.0,z_0,1.0,urdf)
z, y, u = robot.ffd_simulation(w_cp)








    



