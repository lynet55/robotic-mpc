import casadi as ca

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