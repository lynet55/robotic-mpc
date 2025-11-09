import numpy as np
import pinocchio as pin
from pathlib import Path

# This module implements the state space robot model, taking as reference the
# ur5 6 d.o.f industrial manipulator, using different simulation approaches.

class Robot:

    ROOT = Path(__file__).parent
    URDF_DIR = ROOT / "./ur_description/urdf"
    
    def __init__(self, Ts, T, Tu, z0, u_step, urdf_file_name):
        self._Ts = Ts
        self._T = T
        self._Tu = Tu
        self._z0 = z0
        self._u_step = u_step
        self._urdf_file_path = Robot.URDF_DIR / urdf_file_name
        self._mesh_dir = [str(Robot.ROOT)]

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


robot = Robot(0.01, 10.0, 5.0, np.zeros(12), np.zeros(6), "ur5.urdf")