import casadi as ca
import numpy as np
class Surface:

    def __init__(self, position, orientation_rpy, limits):
        self.position = position
        self.orientation_rpy = orientation_rpy
        self.limits = limits
        self.desired_offset = 1.0

        self.x = ca.SX.sym("x")
        self.y = ca.SX.sym("y")
        # a, b, c, d, e, f = 1.0, 0.5, 0.2, -0.3, 0.7, 0.0
        # self.quadratic_surface = a*self.x**2 + b*self.y**2 + c*self.x*self.y + d*self.x + e*self.y + f

        wave_amplitude = 0.1
        wave_freq_x = 20.0
        wave_freq_y = 20.0
        self.quadratic_surface = wave_amplitude * (ca.sin(wave_freq_x * self.x) + ca.sin(wave_freq_y * self.y))


    def get_surface_height(self, x, y):
        return self.quadratic_surface(x, y)

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
        return ca.Function("surface", [self.x, self.y], [self.quadratic_surface])

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
        x_rel = x_surface #Consider Clamping trhese to surface limits
        y_rel = y_surface
        z_rel = float(surface_func(x_rel, y_rel))
        point_local = np.array([x_rel, y_rel, z_rel])
        point_world = self.surface_to_world_transform(point_local)
        
        return (point_local, point_world)

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
        """
        Create a CasAdi function that computes RPY angles at any point on the surface.
        The RPY represents the orientation where the z-axis aligns with the surface normal.
        
        Returns:
            CasAdi Function that takes (x, y) and returns [roll, pitch, yaw]
        """
        # Get the normal vector
        dz_dx = ca.jacobian(self.quadratic_surface, self.x)
        dz_dy = ca.jacobian(self.quadratic_surface, self.y)
        
        # Normal vector components (unnormalized for efficiency)
        nx = -dz_dx
        ny = -dz_dy
        nz = 1.0
        
        # Normalize
        norm = ca.sqrt(nx**2 + ny**2 + nz**2)
        nx_norm = nx / norm
        ny_norm = ny / norm
        nz_norm = nz / norm
        
        # Convert normal vector to RPY angles
        # The normal vector becomes the z-axis of the local frame
        # pitch = asin(-nx_norm)  (rotation about y-axis)
        # roll = atan2(ny_norm, nz_norm)  (rotation about x-axis)
        # yaw = 0  (no rotation about z-axis, arbitrary choice)
        
        pitch = ca.asin(-nx_norm)
        roll = ca.atan2(ny_norm, nz_norm)
        yaw = 0.0
        
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
