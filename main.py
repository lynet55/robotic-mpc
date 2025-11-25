
from turtle import delay, position
import casadi as ca
import numpy as np
import math
import time
from acados import MPC as model_predictive_control
from loader import UrdfLoader as urdf
from visualizer import MeshCatVisualizer as robot_visualizer
from model import Robot as six_dof_model
from model_casadi import SixDofRobot as prediction_model
from surface import Surface

def run_sim(scene, model, mpc, delay_time):
    """
    Run the closed-loop simulation.

    Parameters
    ----------
    scene : MeshCatVisualizer
        Visualizer instance used to render the robot.
    model : SixDofRobot
        Robot model providing dynamics and kinematics.
    solver :
        acados OCP solver instance.
    surface : Surface
        Surface object for task frame positioning.
    surface_origin : np.ndarray
        Origin position of the surface.
    surface_orientation_rpy : np.ndarray
        Orientation of surface in roll-pitch-yaw.
    q_0 : np.ndarray
        Initial joint positions.
    total_time : int
        Number of control steps to simulate.
    delay_time : float, optional
        Delay time between control steps.
    """

    ee_position = model.ee_position(0)
    ee_orientation = model.ee_orientation(0)
    ee_trajectory_points = [ee_position]

    #Scene visualization
    scene.add_line(ee_position.reshape(-1, 3), path="lines/ee_trajectory", color=0xFF0000, line_width=2.0)
    scene.add_line(ee_position.reshape(-1, 3), path="lines/mpc_prediction", color=0x00FF00, line_width=3.0)
    
    # scene.add_line(trajectory.reshape(-1, 3), path="lines/task_reference", color=0x0000FF, line_width=2.0)
    scene.add_triad(
        position=ee_position,
        orientation_rpy=ee_orientation,
        path="frames/end_effector_frame",
        scale=0.2,
        line_width=2.0
    )
    # scene.add_triad(
    #     position=task_origin_world,
    #     orientation_rpy=np.array([0.0, 0.0, 0.0]),
    #     path="frames/task_frame",
    #     scale=0.2,
    #     line_width=2.0
    # )
    # task_xyz_surface, task_xyz_world = surface.get_point_on_surface(surface.limits[0][0] + 5.0 , surface.limits[1][0] + 5.0)
    task_xyz_surface, task_xyz_world = surface.get_random_point_on_surface()
    task_rpy = surface.get_rpy(task_xyz_surface[0], task_xyz_surface[1])

    #Simulation Loop
    for t in range(0, model.N - 1):

        #Random refferences
        if t % 100 == 0:
            task_xyz_surface, task_xyz_world = surface.get_random_point_on_surface()

        # task_normal_surface = surface.get_normal_vector(task_xyz_surface[0], task_xyz_surface[1])

        #MPC input and refferences
        current_state = model.state(t)        
        mpc.solver.set(0, 'lbx', current_state)
        mpc.solver.set(0, 'ubx', current_state)
        status = mpc.solver.solve()
        if status != 0:
            print(f"MPC solver failed with status {status} at t={t}")
            optimal_control = np.zeros(6)  # Fallback to zero control
        else:
            optimal_control = mpc.solver.get(0, "u")

        print(optimal_control)

        # FOR VISUALIZATION, Apply forward and differential kinematics to each predicted state
        # predicted_ee_positions = []
        # predicted_ee_velocities = []
        # for k in range(mpc.ocp.dims.N + 1):
        #     predicted_state = mpc.solver.get(k, "x")
        #     ee_pos = model.forward_kinematics(predicted_state[:6])
        #     ee_vel = model.diff_kinematic(predicted_state[:6], predicted_state[6:])
        #     predicted_ee_positions.append(ee_pos[:3])  # Only position (x, y, z)
        #     predicted_ee_velocities.append(ee_vel)
        
        # predicted_ee_positions = np.array(predicted_ee_positions)
        # predicted_ee_velocities = np.array(predicted_ee_velocities)
      
        #State Update and State Feedback
        model.update(model.state(t), optimal_control, t)
        q1, q2, q3, q4, q5, q6 = model.joint_angles(t)
        
        # Accumulate trajectory
        ee_trajectory_points.append(model.ee_position(t+1))

        #Visual Update
        scene.set_joint_angles({
            "shoulder_pan_joint": q1,
            "shoulder_lift_joint": q2,
            "elbow_joint": q3,
            "wrist_1_joint": q4,
            "wrist_2_joint": q5,
            "wrist_3_joint": q6
        })

        scene.update_triad("frames/end_effector_frame", position=model.ee_position(t+1), orientation_rpy=model.ee_orientation(t+1))
        scene.update_triad("frames/task_frame", position=task_xyz_world, orientation_rpy=task_rpy)
        # scene.update_line("lines/ee_trajectory", points=ee_trajectory_points)
        # scene.update_line("lines/mpc_prediction", points=predicted_ee_positions)
        time.sleep(delay_time)

if __name__ == "__main__":

    surface_limits = ((-0.5, 0.5), (-0.3, 0.3))
    surface_origin = np.array([0.0, 0.0, 0.0])
    surface_orientation_rpy = np.array([0.0, 0.0, 0.0])

    u0 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    z0 = np.zeros([12])
    qdot0 = np.array([2,2,0,0,0,5], dtype=np.float64)

    sample_time = 0.1
    total_time = 1000090000.0   
    dt = sample_time
    N = int(total_time / sample_time)

    robot_loader = urdf('ur5_robot')
    scene = robot_visualizer(robot_loader)
    surface = Surface(
        position=surface_origin,
        orientation_rpy=surface_orientation_rpy,
        limits=surface_limits
    )

    robot = six_dof_model(
        urdf_loader=robot_loader,
        z0=z0,
        u0=u0,
        T=total_time,
        Ts=sample_time,
        wcv=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        integration_method="RK4"
    )

    predictive_model = prediction_model(
        urdf_loader=robot_loader,
        integration_method="RK2"
    )
    predictive_model.set_initial_state(np.hstack((z0[:6], z0[6:])))

    mpc = model_predictive_control(
        surface=surface,
        state=predictive_model.state,
        initial_state=predictive_model.initial_state,
        control_input=predictive_model.control,
        dynamics=predictive_model.get_explicit_model()['ode'],
        forward_kinematics=predictive_model.fk_casadi,  # Uses rotation matrix version
        differential_kinematics=predictive_model.dk_casadi
    )
    
    scene.add_surface_from_casadi(
        surface.get_surface_function(),
        x_limits=surface.get_limits()[0],
        y_limits=surface.get_limits()[1],
        resolution=80,
        path="surfaces/quadratic_surface",
        color=0x3399FF,
        opacity=0.6,    
        origin=surface_origin,             # set position here
        orientation_rpy=surface_orientation_rpy,    # optional roll, pitch, yaw (rad)
    )
    
    run_sim(
        scene,
        robot,
        mpc,
        delay_time = 0.0,
    )