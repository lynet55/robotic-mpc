from loader import UrdfLoader as robotModel
from visualizer import MeshCatVisualizer as robot_visualizer
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
    run_sim(scene, duration = 1000)