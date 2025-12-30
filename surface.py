import casadi as ca
import numpy as np
class Surface:

    def __init__(self, position, orientation_rpy, limits, coefficients=None):
        self.position = position
        self.orientation_rpy = orientation_rpy
        self.limits = limits
        self.desired_offset = 1.0

        self.x = ca.SX.sym("x")
        self.y = ca.SX.sym("y")

        self.coeffs = {
            'a': -0.15, 'b': 0.15, 'c': -0.01,
            'd': 0.01, 'e': 0.01, 'f': 0.0
        }
        if coefficients:
            self.coeffs.update(coefficients)

        self.quadratic_surface = self.coeffs['a']*self.x**2 + self.coeffs['b']*self.y**2 + self.coeffs['c']*self.x*self.y + self.coeffs['d']*self.x + self.coeffs['e']*self.y + self.coeffs['f']
        self._build_surface()

    def _build_surface(self): # |Redundant?|
        """Build the symbolic surface expression from coefficients."""
        c = self.coeffs
        self.quadratic_surface = (
            c['a']*self.x**2 + c['b']*self.y**2 + c['c']*self.x*self.y +
            c['d']*self.x + c['e']*self.y + c['f']
    )

    def rebuild(self):
        """Rebuilds the symbolic surface expression using current coefficients."""
        self.quadratic_surface = self.coeffs['a']*self.x**2 + self.coeffs['b']*self.y**2 + self.coeffs['c']*self.x*self.y + self.coeffs['d']*self.x + self.coeffs['e']*self.y + self.coeffs['f']

    def get_position(self):
        return self.position

    def get_orientation_rpy(self):
        return self.orientation_rpy 

    def get_limits(self):
        return self.limits

    def get_quadratic_surface(self):
        return self.quadratic_surface

    def get_desired_offset(self):
        return self.desired_offset

    def get_surface_function(self):
        return ca.Function("surface", [self.x, self.y], [self.quadratic_surface], ['x','y'], ['S(x,y)'])

    def get_random_point_on_surface(self):
        """
        Generate a random point on the surface in world coordinates.
        Accounts for both surface position and orientation.
        
        Returns:
            numpy array [x, y, z] in world frame
        """
        # Generate random coordinates in local surface frame
        x_rel = np.random.uniform(self.limits[0][0], self.limits[0][1])
        y_rel = np.random.uniform(self.limits[1][0], self.limits[1][1])
        surface_func = self.get_surface_function()
        z_rel = float(surface_func(x_rel, y_rel))
        point_local = np.array([x_rel, y_rel, z_rel])
        point_world = self.surface_to_world_transform(point_local)
        return (point_local, point_world)



    def get_point_on_surface(self, x_surface, y_surface):
        """
        Pass x,y get a z which is on the surface
        Returns:
            numpy array [x, y, z] in world frame
        """
        surface_func = self.get_surface_function()
        # Generate random coordinates in local surface frame
        z = float(surface_func(x_surface, y_surface))
        return z

    def surface_to_world_transform(self, point_surface_frame):
        roll, pitch, yaw = self.orientation_rpy
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: Rz * Ry * Rx (same order as in acados.py)
        R = Rz @ Ry @ Rx

        return R @ point_surface_frame + self.position
      
    def get_rpy_function(self):

        # Partial derivatives
        dS_dx = ca.jacobian(self.quadratic_surface, self.x)
        dS_dy = ca.jacobian(self.quadratic_surface, self.y)

        # Unnormalized normal
        nx = dS_dx
        ny = dS_dy
        nz = -1.0

        # Normalize
        norm = ca.sqrt(nx**2 + ny**2 + nz**2)
        nx = nx / norm
        ny = ny / norm
        nz = nz / norm
        # |CORREZZIONE| Il constraint sull'orientamento del task frame va formulato come task constraint
        #               nella cost function. 
        # Normal vector
        n = ca.vertcat(nx, ny, nz)
        # Global X basis
        ex = ca.vertcat(1.0, 0.0, 0.0)
        # Project global X onto tangent plane to define local X axis
        ex_proj = ex - (ca.dot(ex, n) * n)
        # Normalize local X
        ex_norm = ex_proj / ca.sqrt(ca.dot(ex_proj, ex_proj))
        # Local Y axis = n × ex_norm
        ey_norm = ca.cross(n, ex_norm)
        # Rotation matrix R (columns are local axes)
        R = ca.hcat([ex_norm, ey_norm, n])

        # Extract RPY from rotation matrix assuming XYZ Euler (roll → pitch → yaw)
        # roll
        roll = ca.atan2(R[2,1], R[2,2])
        # pitch
        pitch = -ca.asin(R[2,0])
        # yaw
        yaw = ca.atan2(R[1,0], R[0,0])
        rpy = ca.vertcat(roll, pitch, yaw)
        return ca.Function("surface_rpy", [self.x, self.y], [rpy])
    
    def get_rpy(self, x_surface, y_surface):
        """
        Get the RPY angles at a specific point on the surface.
        
        Args:
            x_surface: x coordinate in surface frame
            y_surface: y coordinate in surface frame
            
        Returns:
            numpy array [roll, pitch, yaw] in radians
        """
        rpy_func = self.get_rpy_function()
        rpy_result = rpy_func(x_surface, y_surface)
        return np.array(rpy_result).flatten()
    
    def generate_simple_trajectory(self, initial_point_task_surface, time_increment, x_margin_surface, y_margin_surface, num_points_x=400, num_points_y=400, x_step=1, y_step=1):
        """
        Generate a serpentine trajectory over the surface grid.
        
        The path:
        - Uses linspace-like grids for x and y within the surface limits.
        - Steps "down" (toward lower y) by one grid index.
        - Reverses x direction and repeats.
        
        Args:
            initial_point_task_surface: iterable of length 2 (x, y) in the surface frame.
            time_increment: unused placeholder for future time-parameterization.
            x_margin_surface: integer margin (grid points) from x boundaries.
            y_margin_surface: integer margin (grid points) from lower y boundary.
            num_points_x: number of grid points along x.
            num_points_y: number of grid points along y.
            x_step: grid index increment when moving along x each step.
            y_step: grid index decrement when stepping down each row along y.
        
        Returns:
            np.ndarray of shape (N, 3) with [x, y, z] points in world coordinates.
        """
        x_min, x_max = self.limits[0]
        y_min, y_max = self.limits[1]
        
        x_coords = np.linspace(x_min, x_max, num=num_points_x)
        y_coords = np.linspace(y_min, y_max, num=num_points_y)
   
        margin_x = x_margin_surface if num_points_x > 2 * x_margin_surface + 1 else max(1, (num_points_x - 1) // 4)
        margin_y = y_margin_surface if num_points_y > 2 * y_margin_surface + 1 else max(1, (num_points_y - 1) // 4)
        
        x0 = float(initial_point_task_surface[0])
        y0 = float(initial_point_task_surface[1])
        
        ix = int(np.argmin(np.abs(x_coords - x0)))
        iy = int(np.argmin(np.abs(y_coords - y0)))
        
        # Clamp start indices to respect the margins
        ix = int(np.clip(ix, x_margin_surface, num_points_x - x_margin_surface - 1))
        iy = int(np.clip(iy, y_margin_surface, num_points_y - y_margin_surface - 1))
        
        # Choose initial x direction based on which half we start in
        x_dir = -1 if ix > (num_points_x // 2) else 1
        
        path_points = []
        path_points.append([x_coords[ix], y_coords[iy]])
        
        # Serpentine scan: move along x within margins, step down in y, reverse x direction
        while iy > margin_y:
            # Traverse along x within the safe margins
            while True:
                next_ix = ix + (x_dir * max(1, int(x_step)))
                if next_ix < margin_x or next_ix > (num_points_x - margin_x - 1):
                    break
                ix = next_ix
                path_points.append([x_coords[ix], y_coords[iy]])
            
            # Step down in y (toward lower boundary)
            next_iy = iy - max(1, int(y_step))
            if next_iy < margin_y:
                break
            iy = next_iy
            path_points.append([x_coords[ix], y_coords[iy]])
            
            # Reverse x direction for the next row
            x_dir *= -1
        
        # Map local (x, y) waypoints to world-frame [x, y, z] on the surface
        path_points_world = [self.get_point_on_surface(x, y)[1] for x, y in path_points]
        
        return np.array(path_points_world)

    def get_normal_vector(self, x, y):
        normal_func = self.get_normal_vector_casadi()
        nx_val, ny_val, nz_val = normal_func(x,y)
        return np.array([nx_val, ny_val, nz_val])
    
    def get_normal_vector_casadi(self):
        # Assume self.quadratic_surface is expressed in terms of x_sym, y_sym
        # Compute partial derivatives symbolically
        dz_dx = ca.jacobian(self.quadratic_surface, self.x)
        dz_dy = ca.jacobian(self.quadratic_surface, self.y)
        
        # Normal vector components
        nx = dz_dx
        ny = dz_dy
        nz = -1.0
        
        # Normalize
        norm = ca.sqrt(nx**2 + ny**2 + nz**2)
        nx_norm = nx / norm
        ny_norm = ny / norm
        nz_norm = nz / norm

        n = ca.vertcat(nx_norm, ny_norm, nz_norm)
        
        return ca.Function("surface", [self.x, self.y], [n], ['x','y'], ['n'])