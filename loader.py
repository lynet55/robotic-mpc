import pinocchio as pin
from pathlib import Path
from typing import Optional, List
from pinocchio.visualize import MeshcatVisualizer

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
            print(f"URDF successfully loaded: {self._urdf_file_path}")
            print(f"nq = {self._model.nq}, ngeoms(vis) = {self._vmodel.ngeoms}, ngeoms(col) = {self._cmodel.ngeoms}")
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
    

class RobotVisualizer:

    def __init__(self, urdf_loader):
        if urdf_loader.model is None:
            raise ValueError("UrdfLoader model is None. Model failed to load.")
        
        self._model = urdf_loader.model
        self._data = urdf_loader.data
        self._q = pin.neutral(self._model)
        
        self._viz = MeshcatVisualizer(
            urdf_loader.model,
            urdf_loader.collision_model,
            urdf_loader.visual_model
        )
        self._viewer_initialized = False
    
    @property
    def model(self):
        return self._model
    
    @property
    def data(self):
        return self._data
    
    @property
    def configuration(self):
        return self._q.copy()
    
    @property
    def joint_names(self):
        return [self._model.names[i] for i in range(1, self._model.njoints)]
    
    def to_browser(self):
        if not self._viewer_initialized:
            self._viz.initViewer(open=True)
            self._viz.loadViewerModel()
            self._viz.display(self._q)
            self._viewer_initialized = True
        return self._viz.viewer.url()
    
    def to_jupyter(self):
        if not self._viewer_initialized:
            self._viz.initViewer(open=False)
            self._viz.loadViewerModel()
            self._viz.display(self._q)
            self._viewer_initialized = True
        return self._viz.viewer.url()
    
    def set_joint_angles(self, angles):
        for joint_name, angle in angles.items():
            if joint_name in self._model.names:
                joint_id = self._model.getJointId(joint_name)
                joint = self._model.joints[joint_id]
                idx_q = joint.idx_q
                
                if joint.nq == 1:
                    self._q[idx_q] = angle
        
        if self._viewer_initialized:
            self._viz.display(self._q)
    
    def set_configuration(self, q):
        if len(q) != self._model.nq:
            raise ValueError(f"Configuration size mismatch")
        self._q = q.copy()
        if self._viewer_initialized:
            self._viz.display(self._q)


# Example usage
if __name__ == "__main__":
    import time
    import math
    
    # Load and visualize
    loader = UrdfLoader("ur5")
    viz = RobotVisualizer(loader)
    
    # Open in browser
    viz.to_browser()
    
    # Animate joints
    for i in range(100):
        viz.set_joint_angles({
            "shoulder_pan_joint": math.sin(i * 0.05),
            "shoulder_lift_joint": math.cos(i * 0.05) * 0.5,
            "elbow_joint": math.sin(i * 0.05) * 0.8,
        })
        time.sleep(0.05)
    
    input("Press Enter to close...")
