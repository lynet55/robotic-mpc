
from turtle import delay, position
import casadi as ca
import numpy as np
import math
import time
from acados import MPC as model_predictive_control
from loader import UrdfLoader as urdf
from visualizer import MeshCatVisualizer as robot_visualizer
# from model import SixDofRobot as six_dof_model
from model_casadi import SixDofRobot as six_dof_model
from surface import Surface

def run_sim(scene, model, mpc, total_time, delay_time: float = 1.0):
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

    q = np.zeros((total_time, model.n_dof)) # joint positions
    q_dot = np.zeros((total_time, model.n_dof)) # joint velocities

    ee_pose_world = np.zeros((total_time, 6)) # end-effector pose [x, y, z, roll, pitch, yaw]
    ee_velocity_world = np.zeros((total_time, 6)) # end-effector velocity [q_dot_x, qdot_y, qdot_z, wx, wy, wz]

    solver = mpc.solver
    N = mpc.ocp.dims.N

    # task_origin_surface, task_origin_world = surface.get_random_point_on_surface()
    task_origin_surface, task_origin_world = surface.get_point_on_surface(surface.limits[0][0] + 5.0 , surface.limits[1][0] + 5.0)
    trajectory_points_surface = surface.generate_simple_trajectory(task_origin_surface, time_increment=0.10, x_margin_surface=5, y_margin_surface=5, x_step=10, y_step=5)
    initial_task_orientation = np.array([0.0, 0.0, 0.0])
    
    # Data structures
    q[0] = model.initial_state[:model.n_dof]
    q_dot[0] = model.initial_state[model.n_dof:]
    ee_pose_world[0] = np.array(model.forward_kinematics(q[0])).flatten()
    ee_velocity_world[0] = np.array(model.differential_kinematics(q[0], q_dot[0])).flatten()


    #Scene visualization
    scene.add_line( np.array(ee_pose_world[0][:3]).reshape(-1, 3), path="lines/ee_trajectory", color=0xFF0000, line_width=2.0)
    scene.add_line(trajectory_points_surface, path="lines/task_reference", color=0x0000FF, line_width=2.0)
    scene.add_triad(
        position=surface.get_position(),
        orientation_rpy=surface.get_orientation_rpy(),
        path="frames/surface_origin",
        scale=0.2,
        line_width=1.0
    )
    scene.add_triad(
        position=np.array([1.0, 0.5, 0.3]),
        orientation_rpy=np.array([0.0, 0.0, 0.0]),
        path="frames/end_effector_frame",
        scale=0.2,
        line_width=2.0
    )
    scene.add_triad(
        position=task_origin_world,
        orientation_rpy=initial_task_orientation,
        path="frames/task_frame",
        scale=0.2,
        line_width=2.0
    )
        
    #Control Loop
    max_traj_idx = len(trajectory_points_surface) - 1
    for t in range(min(total_time - 1, max_traj_idx)):

        #State Feedback and Refferences
        current_q = q[t]
        current_q_dot = q_dot[t]
        current_ee_pose_world = ee_pose_world[t]
        current_ee_velocity_world = ee_velocity_world[t]

        optimal_control = np.zeros(model.n_dof)


        if t % 100 == 0:
            task_xyz_surface = trajectory_points_surface[t]

        # task_velocity_xyz_surface = 1.0, 1.0, 1.0
        task_normal_surface = surface.get_normal_vector(task_xyz_surface[0], task_xyz_surface[1])
        control_input = np.zeros(6)

        #MPC refferences
        current_state = np.concatenate((current_q, current_q_dot)) #[q, q_dot] * 6 i.e joint positions and velocities for 6 DOF robot
        tracking_variables = np.concatenate((
            task_xyz_surface, #constraints for positions
            task_normal_surface, #constraints for normal vector
        ))
        solver.set(0, 'lbx', current_state)
        solver.set(0, 'ubx', current_state)

        for k in range(N + 1):
            solver.set(k, "p", tracking_variables)
        status = solver.solve()
        
        if status != 0:
            if t % 50 == 0:
                print(f"MPC warning: solver status {status} at t={t}; using fallback control")
            optimal_control = np.zeros(model.n_dof)
        else:
            optimal_control = solver.get(0, "u")

        #State Update and State Feedback
        next_state = model.update(current_state, optimal_control)
        q1, q2, q3, q4, q5, q6 = next_state[:model.n_dof]

        #Visual Update
        scene.set_joint_angles({
            "shoulder_pan_joint": q1,
            "shoulder_lift_joint": q2,
            "elbow_joint": q3,
            "wrist_1_joint": q4,
            "wrist_2_joint": q5,
            "wrist_3_joint": q6
        })
        scene.update_triad("frames/end_effector_frame", position=ee_pose_world[t][:3], orientation_rpy=ee_pose_world[t][3:6])
        scene.update_triad("frames/task_frame", position=task_xyz_surface, orientation_rpy=surface.get_rpy(task_xyz_surface[0], task_xyz_surface[1]))
        scene.update_line("lines/ee_trajectory", points=np.array(ee_pose_world[t][:3]).reshape(-1, 3))
        
        q[t + 1] = next_state[:model.n_dof]
        q_dot[t + 1] = next_state[model.n_dof:]
        ee_pose_world[t + 1] = np.array(model.forward_kinematics(q[t + 1])).flatten()
        ee_velocity_world[t + 1] = np.array(model.differential_kinematics(q[t + 1], q_dot[t + 1])).flatten()

        if (t + 1) % 10 == 0:
            print(f"time:  {t}")

        time.sleep(delay_time)
    
    print("\nSimulation complete!")

  

if __name__ == "__main__":

    surface_limits = ((-0.5, 0.5), (-0.3, 0.3))
    # surface_origin = np.array([-0.5, 1.5, 0.2])
    surface_origin = np.array([0.0, 0.0, 0.0])
    # surface_orientation_rpy = np.array([0.9, 0.0, 0.4])
    surface_orientation_rpy = np.array([0.0, 0.0, 0.0])

    qdot_0 = np.array([2,2,0,0,0,5], dtype=np.float64)
    q_0 = np.zeros([6])

    robot_loader = urdf("ur5")
    scene = robot_visualizer(robot_loader)
    surface = Surface(
        position=surface_origin,
        orientation_rpy=surface_orientation_rpy,
        limits=surface_limits
    )
    robot = six_dof_model(
        urdf_loader=robot_loader,
        integration_method="RK2"
    )

    robot.set_initial_state(np.hstack((q_0, qdot_0)))

    mpc = model_predictive_control(
        surface=surface,
        state=robot.state,
        initial_state=robot.initial_state,
        control_input=robot.control,
        dynamics=robot.get_explicit_model()['ode'],
        forward_kinematics=robot.fk_casadi,  # Uses rotation matrix version
        differential_kinematics=robot.dk_casadi
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
        total_time=100000,
        delay_time = 0.01,
    )