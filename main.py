
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

def run_sim(scene, model, solver, total_time, delay_time: float = 1.0):
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

    # t = np.linspace(0, total_time, num=total_time)
    q = np.zeros((total_time, model.n_dof)) # joint positions
    q_dot = np.zeros((total_time, model.n_dof)) # joint velocities

    end_effector_pose = np.zeros((total_time, 7)) # end-effector pose [x, y, z, qx, qy, qz, qw]
    end_effector_velocity = np.zeros((total_time, 6)) # end-effector velocity [vx, vy, vz, wx, wy, wz]

    # Set initial conditions from robot's initial state
    q[0] = model.initial_state[:model.n_dof]
    q_dot[0] = model.initial_state[model.n_dof:]
    end_effector_pose[0] = np.array(model.forward_kinematics(q[0])).flatten()
    end_effector_velocity[0] = np.array(model.differential_kinematics(q[0], q_dot[0])).flatten()
    
    # Initialize trajectory tracking - KEEP AS LIST
    trajectory_points = [end_effector_pose[0][:3].tolist()]

    task_origin_local, task_origin_world = surface.get_random_point_on_surface() #initial task origin - returns (local, world)
    initial_task_orientation = np.array([0.0, 0.0, 0.0])

    initial_ee_pos = end_effector_pose[0][:3]
    
    # Add coordinate frame triad at surface origin
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
        line_width=1.0
    )
    scene.add_triad(
        position=task_origin_world, #Initial task origin in world frame
        orientation_rpy=initial_task_orientation,
        path="frames/task_frame",
        scale=0.2,
        line_width=1.0
    )

    # Initialize trajectory line with initial position
    scene.add_line(np.array([initial_ee_pos]).reshape(-1, 3), path="lines/trajectory", color=0xFF0000, line_width=2.0)
    
    print(f"Starting simulation for {total_time} steps...")
    print(f"Initial end-effector position: {end_effector_pose[0][:3]}")
    
    #Control Loop
    for t in range(total_time - 1):

        current_q = q[t]
        current_q_dot = q_dot[t]
        current_ee_pose = end_effector_pose[t]
        current_ee_velocity = end_effector_velocity[t]

        # Set current constraint: [q, q_dot]
        current_state = np.concatenate((current_q, current_q_dot))
        solver.set(0, 'lbx', current_state)
        solver.set(0, 'ubx', current_state)

        status = solver.solve()
        optimal_control = solver.get(0, "u")

        # Update the full state using the integrator
        next_state = model.update(current_state, optimal_control)
        q1, q2, q3, q4, q5, q6 = next_state[:model.n_dof]

        scene.set_joint_angles({
            "shoulder_pan_joint": q1,
            "shoulder_lift_joint": q2,
            "elbow_joint": q3,
            "wrist_1_joint": q4,
            "wrist_2_joint": q5,
            "wrist_3_joint": q6
        })

        # Extract position and velocity from the integrated state
        q[t + 1] = next_state[:model.n_dof]
        q_dot[t + 1] = next_state[model.n_dof:]
        
        # Compute end-effector kinematics (flatten to 1D arrays)
        end_effector_pose[t + 1] = np.array(model.forward_kinematics(q[t + 1])).flatten()
        end_effector_velocity[t + 1] = np.array(model.differential_kinematics(q[t + 1], q_dot[t + 1])).flatten()

        # Move task frame along x in surface coordinates
        x_new_task_origin_local = task_origin_local[0] - 0.0001
        y_new_task_origin_local = task_origin_local[1] + 0.0001
        task_origin_local, task_origin_world = surface.get_point_on_surface(x_new_task_origin_local, y_new_task_origin_local)

        if (t + 1) % 10 == 0:
            trajectory_points.append(end_effector_pose[t + 1][:3].tolist())
            scene.update_line("lines/trajectory", points=np.array(trajectory_points))
            scene.update_triad("frames/end_effector_frame", position=end_effector_pose[t][:3], orientation_rpy=scene.quaternion_to_euler_numpy(end_effector_pose[t][3:]))
            scene.update_triad("frames/task_frame", position=task_origin_world, orientation_rpy=surface.get_rpy(x_new_task_origin_local, y_new_task_origin_local))


        time.sleep(delay_time)
    
    print("\nSimulation complete!")
    print(f"Final end-effector position: {end_effector_pose[-1][:3]}")

  

if __name__ == "__main__":

    surface_limits = ((-0.5, 0.5), (-0.3, 0.3))
    surface_origin = np.array([-0.5, 1.5, 0.2])
    surface_orientation_rpy = np.array([0.9, 0.0, 0.4])


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

    qdot_0 = np.array([2,2,0,0,0,5], dtype=np.float64)
    q_0 = np.zeros([6])
    robot.set_initial_state(np.hstack((q_0, qdot_0)))

    mpc = model_predictive_control(
        surface=surface,
        state=robot.state,
        initial_state=robot.initial_state,
        control_input=robot.control,
        dynamics=robot.get_explicit_model()['ode'],
        forward_kinematics=robot.fk_casadi,
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
        mpc.solver,
        total_time=100000,
        delay_time = 0.01,
    )