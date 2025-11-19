
from turtle import delay
import casadi as ca
import numpy as np
import math
import time
from acados import MPC as model_predictive_control
from loader import UrdfLoader as urdf
from visualizer import MeshCatVisualizer as robot_visualizer
# from model import SixDofRobot as six_dof_model
from model_casadi import SixDofRobot as six_dof_model


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
    
    # Initialize trajectory tracking
    trajectory_points = [end_effector_pose[0][:3]]
    
    print(f"Starting simulation for {total_time} steps...")
    print(f"Initial end-effector position: {end_effector_pose[0][:3]}")
    
    #Control Loop
    for t in range(total_time - 1):

        current_q = q[t]
        current_q_dot = q_dot[t]

        
        # Set current constraint: [q, q_dot]
        current_state = np.concatenate((current_q, current_q_dot))
        solver.set(0, 'lbx', current_state)
        solver.set(0, 'ubx', current_state)

        status = solver.solve()
        optimal_control = solver.get(0, "u")

        # print("Optimal control: ", optimal_control)

        # Update the full state using the integrator
        next_state = model.update(current_state, optimal_control)

        q1 = next_state[0]
        q2 = next_state[1]
        q3 = next_state[2]
        q4 = next_state[3]
        q5 = next_state[4]
        q6 = next_state[5]

        # print(q)

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

        # Update trajectory line every 10 steps for performance
        if (t + 1) % 10 == 0:
            trajectory_points.append(end_effector_pose[t + 1][:3])
            scene.update_line("lines/trajectory", points=np.array(trajectory_points))

        time.sleep(delay_time)
    
    print("\nSimulation complete!")
    print(f"Final end-effector position: {end_effector_pose[-1][:3]}")

  

if __name__ == "__main__":

    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    a, b, c, d, e, f = 1.0, 0.5, 0.2, -0.3, 0.7, 0.0
    quadratic_surface = a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

    joint_range = [-2*np.pi, 2*np.pi]
    joint_limits = np.array([joint_range, joint_range, joint_range, joint_range, joint_range, joint_range])
    q_0 = np.array([
        np.random.uniform(joint_limits[i, 0], joint_limits[i, 1]) 
        for i in range(6)
    ], dtype=np.float64)  
    
    qdot_0 = np.array([2,2,0,0,0,5], dtype=np.float64) #Initial angular speeds

    robot_loader = urdf("ur5")
    scene = robot_visualizer(robot_loader)
    
    robot = six_dof_model(
        urdf_loader=robot_loader,
        initial_state = np.hstack((q_0, qdot_0)),
        integration_method="RK2"
    )

    # Surface parameters (must match visualization)
    surface_position = np.array([-0.5, 1.5, 0.2])
    surface_orientation_rpy = np.array([0.9, 0.0, 0.4])
    desired_offset = 1.0  # Maintain 1 unit offset above surface
    
    mpc = model_predictive_control(
        surface=quadratic_surface,
        state=robot.state,
        control_input=robot.control,
        dynamics=robot.get_explicit_model()['ode'],
        forward_kinematics=robot.fk_casadi,
        differential_kinematics=robot.dk_casadi,
        surface_position=surface_position,
        surface_orientation_rpy=surface_orientation_rpy,
        desired_offset=desired_offset,
    )

    scene.add_surface_from_casadi(
        quadratic_surface, x, y,
        # x_limits=(-0.5, 0.5),
        # y_limits=(-0.3, 0.3),
        x_limits=(-5, 0.5),
        y_limits=(-3, 0.3),
        resolution=80,
        path="surfaces/quadratic_surface",
        color=0x3399FF,
        opacity=0.6,    
        origin=(-0.5, 1.5, 0.2),             # set position here
        orientation_rpy=(0.9, 0.0, 0.4),    # optional roll, pitch, yaw (rad)
    )

    # Initialize trajectory line for end-effector tracking
    initial_ee_pos = np.array(robot.forward_kinematics(q_0)).flatten()[:3]
    trajectory_points = np.array([initial_ee_pos])
    scene.add_line(trajectory_points.reshape(-1, 3), path="lines/trajectory", color=0xFF0000, line_width=2.0)
    
    run_sim(
        scene,
        robot,
        mpc.solver,
        total_time=100000,
        delay_time = 0.01,
    )