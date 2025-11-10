from loader import UrdfLoader as robotModel
from visualizer import MeshCatVisualizer as robot_visualizer
import casadi as ca
import math
import time

def run_sim(scene, duration):

    for step in range(duration):

        q1 = math.sin(step * 0.05)
        q2 = math.cos(step * 0.05) * 0.5
        q3 = math.cos(step * 0.05) * 0.8
        q4 = math.sin(step * 0.05)
        q5 = math.cos(step * 0.05) * 0.5
        q6 = math.cos(step * 0.2) * 0.8

        scene.set_joint_angles({
            "shoulder_pan_joint": q1,
            "shoulder_lift_joint": q2,
            "elbow_joint": q3,
            "wrist_1_joint": q4,
            "wrist_2_joint": q5,
            "wrist_3_joint": q6
        })

    

if __name__ == "__main__":
    model = robotModel("ur5")
    scene = robot_visualizer(model)

    # Symbols and quadratic expression: a x^2 + b y^2 + c x*y + d x + e + f
    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    a, b, c, d, e, f = 1.0, 0.5, 0.2, -0.3, 0.7, 0.0
    expr = a*x*x + b*y*y + c*x*y + d*x + e + f  # e is constant per your form

    scene.add_surface_from_casadi(
        expr, x, y,
        x_limits=(-0.5, 0.5),
        y_limits=(-0.3, 0.3),
        resolution=80,
        path="surfaces/quadratic_surface",
        color=0x3399FF,
        opacity=0.6,    
        origin=(-0.5, 1.5, 0.2),             # set position here
        orientation_rpy=(0.9, 0.0, 0.4),    # optional roll, pitch, yaw (rad)
    )

    run_sim(scene, duration = 1000)