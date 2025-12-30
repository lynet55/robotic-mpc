import pinocchio as pin
from pathlib import Path
from typing import Optional, List

class UrdfLoader:

    ROOT = Path(__file__).parent
    URDF_DIR = ROOT / "ur_description/urdf"
    
    def __init__(self, robot_name: str) -> None:
        self._urdf_file_path: Path = self.URDF_DIR / f"{robot_name}.urdf"
        self._mesh_dir: List[str] = [str(self.ROOT)]
        self._model: Optional[pin.Model] = None
        self._vmodel = None
        self._cmodel = None
        self._data: Optional[pin.Data] = None
        self._fee: Optional[int] = None
        self.name = robot_name


        try:
            # Load model with both VISUAL and COLLISION geometry types
            # This ensures DAE files are loaded for visuals, STL for collisions
            self._model, self._cmodel, self._vmodel = pin.buildModelsFromUrdf(
                str(self._urdf_file_path),
                package_dirs=self._mesh_dir,
                geometry_types=[pin.GeometryType.COLLISION, pin.GeometryType.VISUAL],
            )
            print(f"URDF successfully loaded: {self._urdf_file_path}")
            print(f"nq = {self._model.nq}, ngeoms(col) = {self._cmodel.ngeoms}, ngeoms(vis) = {self._vmodel.ngeoms}")
            self._data = self._model.createData()
            try:
                if self.name == 'ur5':
                    self._fee = self._model.getFrameId("tool0")
                if self.name == 'ur10':
                    self._fee = self._model.getFrameId("ee_link")
            except Exception as e:
                raise RuntimeError(
                    "Failed to resolve end-effector frame 'tool0' in the URDF model."
                ) from e

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

    @property
    def model(self):
        return self._model
    @property
    def data(self):
        return self._data    
    
    @property
    def collision_model(self):
        return self._cmodel
    
    @property
    def visual_model(self):
        return self._vmodel

    @property
    def fee(self):
        return self._fee  