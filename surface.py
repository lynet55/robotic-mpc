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
        a, b, c, d, e, f = 1.0, 0.5, 0.2, -0.3, 0.7, 0.0
        self.quadratic_surface = a*self.x**2 + b*self.y**2 + c*self.x*self.y + d*self.x + e*self.y + f

    def get_surface_height(self, x_rel, y_rel):
        """
        Get surface height in local surface frame coordinates.
        
        Args:
            x_rel: x coordinate in surface frame
            y_rel: y coordinate in surface frame
        
        Returns:
            z height in surface frame
        """
        surface_func = self.get_surface_function()
        return surface_func(x_rel, y_rel)

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
        
        # Point in local surface frame
        point_local = np.array([x_rel, y_rel, z_rel])
        
        # Create rotation matrix from RPY (Roll-Pitch-Yaw)
        roll, pitch, yaw = self.orientation_rpy
        
        # Rotation matrices
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
        
        # Transform to world coordinates: p_world = R * p_local + position
        point_world = R @ point_local + self.position
        
        return point_world
