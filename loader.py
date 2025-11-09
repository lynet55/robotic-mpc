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

        try:
            self._model, self._vmodel, self._cmodel = pin.buildModelsFromUrdf(
                str(self._urdf_file_path),
                package_dirs=self._mesh_dir,
            )
            self._data = self._model.createData()

        except FileNotFoundError as e:
            print(f"File not found: {e.filename}")

        except ValueError as e:
            print(f"Error while parsing the URDF or building the model:\n{e}")

        except Exception as e:
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