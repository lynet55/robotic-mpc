from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin

class MeshCatVisualizer:

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
        self._viz.initViewer(open=True)
        self._viz.loadViewerModel()
        self._viz.display(self._q)


    def set_joint_angles(self, angles):
        for joint_name, angle in angles.items():
            if joint_name in self._model.names:
                joint_id = self._model.getJointId(joint_name)
                joint = self._model.joints[joint_id]
                idx_q = joint.idx_q
                
                if joint.nq == 1:
                    self._q[idx_q] = angle
        self._viz.display(self._q)