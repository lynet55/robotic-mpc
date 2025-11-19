from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import numpy as np
import casadi as ca
import meshcat.geometry as g
import meshcat.transformations as tf

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
        
        # Hide background by default
        self._viz.viewer["/Background"].set_property("visible", False)
        
        # Storage for lines
        self._lines = {}

    def jupyter_cell(self):
        """Return the Jupyter widget for inline rendering."""
        return self._viz.viewer.jupyter_cell()

    def set_joint_angles(self, angles):
        for joint_name, angle in angles.items():
            if joint_name in self._model.names:
                joint_id = self._model.getJointId(joint_name)
                joint = self._model.joints[joint_id]
                idx_q = joint.idx_q
                
                if joint.nq == 1:
                    self._q[idx_q] = angle
        self._viz.display(self._q)

    def add_surface_from_casadi(
        self,
        expr,
        x_symbol,
        y_symbol,
        x_limits=(-1.0, 1.0),
        y_limits=(-1.0, 1.0),
        resolution=50,
        path="surfaces/quadratic",
        color=0x3399FF,
        opacity=0.6,
        origin=(0.0, 0.0, 0.0),
        orientation_rpy=None,
    ):
        """
        Add a quadratic-like surface defined by a CasADi scalar expression z = f(x, y) to MeshCat.
        
        Parameters
        - expr: CasADi expression (SX or MX) for z given x_symbol and y_symbol
        - x_symbol, y_symbol: CasADi symbols used in expr
        - x_limits, y_limits: tuples defining grid extents
        - resolution: number of samples per axis (>= 2)
        - path: MeshCat path where the surface will be inserted (e.g., 'surfaces/my_surface')
        - color: RGB hex integer (e.g., 0x3399FF)
        - opacity: 0.0-1.0
        - origin: (x, y, z) world translation of the surface origin
        - orientation_rpy: optional (roll, pitch, yaw) in radians for world orientation
        """
        if resolution < 2:
            raise ValueError("resolution must be >= 2")
        
        # Build callable CasADi function f(x, y) -> z
        f = ca.Function("surface", [x_symbol, y_symbol], [expr])
        
        # Create grid
        xs = np.linspace(float(x_limits[0]), float(x_limits[1]), int(resolution))
        ys = np.linspace(float(y_limits[0]), float(y_limits[1]), int(resolution))
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        Z = np.zeros_like(X, dtype=float)
        
        # Evaluate z over grid
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = float(f(X[i, j], Y[i, j]))
        
        # Build vertex array (N x 3) and face indices (M x 3)
        vertices = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        faces = []
        n_x, n_y = X.shape
        for i in range(n_x - 1):
            for j in range(n_y - 1):
                v0 = i * n_y + j
                v1 = v0 + 1
                v2 = v0 + n_y
                v3 = v2 + 1
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        faces = np.asarray(faces, dtype=np.uint32)
        
        # MeshCat expects Nx3 for vertices and Mx3 for faces
        vertices_mc = vertices.astype(np.float32, copy=False)
        faces_mc = faces.astype(np.uint32, copy=False)
        
        geom = g.TriangularMeshGeometry(vertices_mc, faces_mc)
        material = g.MeshLambertMaterial(
            color=int(color),
            opacity=float(opacity),
            transparent=(opacity < 1.0),
        )
        
        # Resolve MeshCat path and set object
        node = self._viz.viewer
        for part in str(path).split("/"):
            if part:
                node = node[part]
        node.set_object(geom, material)
        
        # Apply world pose (translation + optional rotation)
        try:
            T = tf.translation_matrix([float(origin[0]), float(origin[1]), float(origin[2])])
            if orientation_rpy is not None:
                roll, pitch, yaw = map(float, orientation_rpy)
                R = tf.euler_matrix(roll, pitch, yaw)
                T = T @ R
            node.set_transform(T)
        except Exception:
            # Keep surface at world origin if transform cannot be applied
            pass

    def add_line(self, points, path="lines/line", color=0xFF0000, line_width=2.0):
        """
        Add a line to the visualization.
        
        Parameters
        - points: Nx3 numpy array or list of 3D points defining the line segments
        - path: MeshCat path where the line will be inserted (e.g., 'lines/trajectory')
        - color: RGB hex integer (e.g., 0xFF0000 for red)
        - line_width: width of the line
        """
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be an Nx3 array")
        
        # Store line data for later updates
        self._lines[path] = {
            'points': points.copy(),
            'color': color,
            'line_width': line_width
        }
        
        # Create line geometry and material
        geom = g.Line(
            g.PointsGeometry(points.T),
            g.LineBasicMaterial(color=int(color), linewidth=float(line_width))
        )
        
        # Resolve MeshCat path and set object
        node = self._viz.viewer
        for part in str(path).split("/"):
            if part:
                node = node[part]
        node.set_object(geom)

    def update_line(self, path, points=None, color=None, line_width=None):
        """
        Update an existing line in the visualization.
        
        Parameters
        - path: MeshCat path of the line to update
        - points: optional Nx3 numpy array or list of new 3D points
        - color: optional new RGB hex integer
        - line_width: optional new line width
        """
        if path not in self._lines:
            raise ValueError(f"Line at path '{path}' does not exist. Use add_line() first.")
        
        # Update stored data
        if points is not None:
            points = np.asarray(points, dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("points must be an Nx3 array")
            self._lines[path]['points'] = points.copy()
        
        if color is not None:
            self._lines[path]['color'] = color
        
        if line_width is not None:
            self._lines[path]['line_width'] = line_width
        
        # Get current line data
        line_data = self._lines[path]
        
        # Recreate line geometry and material with updated data
        geom = g.Line(
            g.PointsGeometry(line_data['points'].T),
            g.LineBasicMaterial(
                color=int(line_data['color']),
                linewidth=float(line_data['line_width'])
            )
        )
        
        # Resolve MeshCat path and update object
        node = self._viz.viewer
        for part in str(path).split("/"):
            if part:
                node = node[part]
        node.set_object(geom)

    def add_point(self, position, path="points/point", color=0xFF0000, radius=0.02, opacity=1.0):
        """
        Add a point (sphere) to the visualization.
        
        Parameters
        - position: 3-element array or list [x, y, z] specifying the point location
        - path: MeshCat path where the point will be inserted (e.g., 'points/target')
        - color: RGB hex integer (e.g., 0xFF0000 for red)
        - radius: radius of the sphere representing the point
        - opacity: opacity of the point (0.0-1.0)
        """
        position = np.asarray(position, dtype=np.float32)
        if position.shape != (3,):
            raise ValueError("position must be a 3-element array [x, y, z]")
        
        # Create sphere geometry and material
        geom = g.Sphere(float(radius))
        material = g.MeshLambertMaterial(
            color=int(color),
            opacity=float(opacity),
            transparent=(opacity < 1.0)
        )
        
        # Resolve MeshCat path and set object
        node = self._viz.viewer
        for part in str(path).split("/"):
            if part:
                node = node[part]
        node.set_object(geom, material)
        
        # Set position
        T = tf.translation_matrix([float(position[0]), float(position[1]), float(position[2])])
        node.set_transform(T)