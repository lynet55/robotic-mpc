from pinocchio.visualize import MeshcatVisualizer as PinMeshcatVisualizer
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
        
        self._viz = PinMeshcatVisualizer(
            urdf_loader.model,
            urdf_loader.collision_model,
            urdf_loader.visual_model
        )
        self._viz.initViewer(open=True)
        self._viz.loadViewerModel()
        #self._add_lights()
        self._viz.displayVisuals(True)
        self._viz.displayCollisions(False)
        self._viz.display(self._q)
        
        # Store collision model reference
        self._collision_model = urdf_loader.collision_model
        
        # Add ambient light to properly illuminate collision geometries
        self._viz.viewer["/Lights/AmbientLight"].set_property("intensity", 0.8)
        
        # Apply proper materials to collision geometries after loading
        self._apply_collision_materials(color=0xFF8000, opacity=0.5)
        
        # Hide background by default
        self._viz.viewer["/Background"].set_property("visible", False)
        
        # Storage for lines
        self._lines = {}



    def _add_lights(self):
        # 1) Point light
        try:
            self._viz.viewer["/Lights/PointLight"].set_object(
                g.PointLight(color=0xFFFFFF, intensity=1.0)
            )
            self._viz.viewer["/Lights/PointLight"].set_transform(
                tf.translation_matrix([2.0, 2.0, 2.0])
            )
        except Exception as e:
            print("[MeshCat] Could not add PointLight:", e)

        # 2) Directional light (optional)
        try:
            self._viz.viewer["/Lights/DirectionalLight"].set_object(
                g.DirectionalLight(color=0xFFFFFF, intensity=1.0)
            )
            self._viz.viewer["/Lights/DirectionalLight"].set_transform(
                tf.translation_matrix([3.0, 3.0, 3.0])
            )
        except Exception as e:
            print("[MeshCat] Could not add DirectionalLight:", e)


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
        casadi_surface_function,
        x_limits=(-1.0, 1.0),
        y_limits=(-1.0, 1.0),
        resolution=50,
        path="surfaces/quadratic",
        color=0x3399FF,
        opacity=0.6,
        origin=None,
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
        - origin: numpy array [x, y, z] for world translation of the surface origin
        - orientation_rpy: optional numpy array [roll, pitch, yaw] in radians for world orientation
        """
        if resolution < 2:
            raise ValueError("resolution must be >= 2")
        
        # Validate origin is a numpy array
        if origin is None:
            origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif not isinstance(origin, np.ndarray) or origin.shape != (3,):
            raise TypeError("origin must be a numpy array with shape (3,)")
        
        # Validate orientation_rpy is a numpy array if provided
        if orientation_rpy is not None:
            if not isinstance(orientation_rpy, np.ndarray) or orientation_rpy.shape != (3,):
                raise TypeError("orientation_rpy must be a numpy array with shape (3,)")
        
        # Build callable CasADi function f(x, y) -> z
        f = casadi_surface_function
        
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
                roll, pitch, yaw = float(orientation_rpy[0]), float(orientation_rpy[1]), float(orientation_rpy[2])
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
        - points: Nx3 numpy array of 3D points defining the line segments
        - path: MeshCat path where the line will be inserted (e.g., 'lines/trajectory')
        - color: RGB hex integer (e.g., 0xFF0000 for red)
        - line_width: width of the line
        """
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
            raise TypeError("points must be a numpy array with shape (N, 3)")
        
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

    def add_dashed_isoline_on_surface(
        self,
        casadi_surface_function,
        x_const,
        y_limits,
        n_samples=400,
        dash_every=12,         
        path="lines/px_ref",
        color=0xD97706,
        line_width=3.0,
        origin=None,
        orientation_rpy=None,
    ):
        """
        Draw a dashed isoline on the surface: x = x_const, y in [ymin, ymax], z = f(x, y).
        Dashed effect is approximated by alternating small segments and gaps.
        """

        # sample along y in surface frame
        ys = np.linspace(float(y_limits[0]), float(y_limits[1]), int(n_samples))
        pts = np.zeros((n_samples, 3), dtype=np.float32)

        for i, y in enumerate(ys):
            z = float(casadi_surface_function(float(x_const), float(y)))
            pts[i, :] = np.array([float(x_const), float(y), float(z)], dtype=np.float32)

        # apply same world transform used for the surface (origin + rpy), so the line sits on it
        if origin is None:
            origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if orientation_rpy is None:
            orientation_rpy = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        roll, pitch, yaw = float(orientation_rpy[0]), float(orientation_rpy[1]), float(orientation_rpy[2])
        T = tf.translation_matrix([float(origin[0]), float(origin[1]), float(origin[2])]) @ tf.euler_matrix(roll, pitch, yaw)

        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
        pts_w = (T @ pts_h.T).T[:, :3]

        # build dashed segments: take pairs (i, i+1) only on "on" intervals
        seg_points = []
        on = True
        counter = 0
        for i in range(len(pts_w) - 1):
            if on:
                seg_points.append(pts_w[i])
                seg_points.append(pts_w[i + 1])
            counter += 1
            if counter >= int(dash_every):
                counter = 0
                on = not on

        if len(seg_points) == 0:
            return

        seg_points = np.array(seg_points, dtype=np.float32)  # shape (2*M, 3)

        geom = g.LineSegments(
            g.PointsGeometry(seg_points.T),
            g.LineBasicMaterial(color=int(color), linewidth=float(line_width))
        )

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
        - points: optional Nx3 numpy array of new 3D points
        - color: optional new RGB hex integer
        - line_width: optional new line width
        """
        if path not in self._lines:
            raise ValueError(f"Line at path '{path}' does not exist. Use add_line() first.")
        
        # Update stored data
        if points is not None:
            if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
                raise TypeError("points must be a numpy array with shape (N, 3)")
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
        - position: numpy array [x, y, z] specifying the point location
        - path: MeshCat path where the point will be inserted (e.g., 'points/target')
        - color: RGB hex integer (e.g., 0xFF0000 for red)
        - radius: radius of the sphere representing the point
        - opacity: opacity of the point (0.0-1.0)
        """
        if not isinstance(position, np.ndarray) or position.shape != (3,):
            raise TypeError("position must be a numpy array with shape (3,)")
        
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

    def add_triad(self, position, orientation_rpy=None, path="frames/triad", scale=0.1, line_width=3.0):
        """
        Add a coordinate frame triad (X, Y, Z axes) to the visualization.
        
        Parameters
        - position: numpy array [x, y, z] specifying the triad origin
        - orientation_rpy: optional numpy array [roll, pitch, yaw] in radians for triad orientation
        - path: MeshCat path where the triad will be inserted (e.g., 'frames/surface_origin')
        - scale: length of each axis arrow
        - line_width: width of the axis lines
        """
        if not isinstance(position, np.ndarray) or position.shape != (3,):
            raise TypeError("position must be a numpy array with shape (3,)")
        
        if orientation_rpy is not None:
            if not isinstance(orientation_rpy, np.ndarray) or orientation_rpy.shape != (3,):
                raise TypeError("orientation_rpy must be a numpy array with shape (3,)")
        
        # Define axis colors: X=red, Y=green, Z=blue
        axes_config = [
            ('x', np.array([scale, 0.0, 0.0]), 0xFF0000),
            ('y', np.array([0.0, scale, 0.0]), 0x00FF00),
            ('z', np.array([0.0, 0.0, scale]), 0x0000FF),
        ]
        
        for axis_name, axis_direction, color in axes_config:
            # Create line from origin to axis endpoint
            points = np.array([[0.0, 0.0, 0.0], axis_direction], dtype=np.float32)
            
            geom = g.Line(
                g.PointsGeometry(points.T),
                g.LineBasicMaterial(color=int(color), linewidth=float(line_width))
            )
            
            # Resolve MeshCat path for this axis
            axis_path = f"{path}/{axis_name}"
            node = self._viz.viewer
            for part in str(axis_path).split("/"):
                if part:
                    node = node[part]
            node.set_object(geom)
            
            # Apply transform (translation + rotation)
            T = tf.translation_matrix([float(position[0]), float(position[1]), float(position[2])])
            if orientation_rpy is not None:
                roll, pitch, yaw = float(orientation_rpy[0]), float(orientation_rpy[1]), float(orientation_rpy[2])
                R = tf.euler_matrix(roll, pitch, yaw)
                T = T @ R
            node.set_transform(T)
    
    def update_point(self, path, position=None, color=None, radius=None, opacity=None):
        """
        Update an existing point in the visualization.
        
        Parameters
        - path: MeshCat path of the point to update
        - position: optional numpy array [x, y, z] for new position
        - color: optional new RGB hex integer
        - radius: optional new radius
        - opacity: optional new opacity (0.0-1.0)
        """
        # Validate position if provided
        if position is not None:
            if not isinstance(position, np.ndarray) or position.shape != (3,):
                raise TypeError("position must be a numpy array with shape (3,)")
        
        # Resolve MeshCat path
        node = self._viz.viewer
        for part in str(path).split("/"):
            if part:
                node = node[part]
        
        # If geometry/material properties changed, recreate the object
        if color is not None or radius is not None or opacity is not None:
            # Use current values as defaults if not provided
            current_radius = radius if radius is not None else 0.02
            current_color = color if color is not None else 0xFF0000
            current_opacity = opacity if opacity is not None else 1.0
            
            geom = g.Sphere(float(current_radius))
            material = g.MeshLambertMaterial(
                color=int(current_color),
                opacity=float(current_opacity),
                transparent=(current_opacity < 1.0)
            )
            node.set_object(geom, material)
        
        # Update position if provided
        if position is not None:
            T = tf.translation_matrix([float(position[0]), float(position[1]), float(position[2])])
            node.set_transform(T)

    def update_triad(self, path, position=None, orientation_rpy=None):
        """
        Update an existing triad's position and/or orientation.
        
        Parameters
        - path: MeshCat path of the triad to update
        - position: optional numpy array [x, y, z] for new position
        - orientation_rpy: optional numpy array [roll, pitch, yaw] for new orientation
        """
        # Validate inputs if provided
        if position is not None:
            if not isinstance(position, np.ndarray) or position.shape != (3,):
                raise TypeError("position must be a numpy array with shape (3,)")
        
        if orientation_rpy is not None:
            if not isinstance(orientation_rpy, np.ndarray) or orientation_rpy.shape != (3,):
                raise TypeError("orientation_rpy must be a numpy array with shape (3,)")
        
        # If no updates requested, return early
        if position is None and orientation_rpy is None:
            return
        
        # Update each axis of the triad (x, y, z)
        for axis_name in ['x', 'y', 'z']:
            axis_path = f"{path}/{axis_name}"
            node = self._viz.viewer
            for part in str(axis_path).split("/"):
                if part:
                    node = node[part]
            
            # Build new transform
            # Use provided position or keep existing (would need to track state for full implementation)
            # For simplicity, we require position to be provided
            if position is None:
                raise ValueError("position must be provided to update triad (tracking existing state not implemented)")
            
            T = tf.translation_matrix([float(position[0]), float(position[1]), float(position[2])])
            if orientation_rpy is not None:
                roll, pitch, yaw = float(orientation_rpy[0]), float(orientation_rpy[1]), float(orientation_rpy[2])
                R = tf.euler_matrix(roll, pitch, yaw)
                T = T @ R
            
            node.set_transform(T)
    