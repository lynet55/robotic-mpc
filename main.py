import casadi as ca
import numpy as np
import math

from acados import MPC as model_predictive_control
from loader import UrdfLoader as urdf
from visualizer import MeshCatVisualizer as robot_visualizer
# from model import SixDofRobot as six_dof_model
from model_casadi import SixDofRobot as six_dof_model


def run_sim(scene, model, solver, total_time):

    time = np.linspace(0, total_time, num=total_time)
    q = np.zeros((total_time, model.n_dof)) # joint positions
    q_dot = np.zeros((total_time, model.n_dof)) # joint velocities

    end_effector_pose = np.zeros((total_time, 7)) # end-effector pose [x, y, z, qx, qy, qz, qw]
    end_effector_velocity = np.zeros((total_time, 6)) # end-effector velocity [vx, vy, vz, wx, wy, wz]

    # Set initial conditions from robot's initial state
    q[0] = model.initial_state[:model.n_dof]
    q_dot[0] = model.initial_state[model.n_dof:]
    end_effector_pose[0] = np.array(model.forward_kinematics(q[0])).flatten()
    end_effector_velocity[0] = np.array(model.differential_kinematics(q[0], q_dot[0])).flatten()
    
    print(f"Starting simulation for {total_time} steps...")
    print(f"Initial end-effector position: {end_effector_pose[0][:3]}")
    
    #Control Loop
    for t in range(total_time - 1):

        current_q = q[t]
        current_q_dot = q_dot[t]
        
        # Set initial state constraint: [q, q_dot]
        current_state = np.concatenate((current_q, current_q_dot))
        solver.set(0, 'lbx', current_state)
        solver.set(0, 'ubx', current_state)

        status = solver.solve()
        optimal_control = solver.get(0, "u")

        # Update the full state using the integrator
        next_state = model.update(current_state, optimal_control)
        
        # Extract position and velocity from the integrated state
        q[t + 1] = next_state[:model.n_dof]
        q_dot[t + 1] = next_state[model.n_dof:]
        
        # Compute end-effector kinematics (flatten to 1D arrays)
        end_effector_pose[t + 1] = np.array(model.forward_kinematics(q[t + 1])).flatten()
        end_effector_velocity[t + 1] = np.array(model.differential_kinematics(q[t + 1], q_dot[t + 1])).flatten()

        q1 = math.sin(t * 0.05)
        q2 = math.cos(t * 0.05) * 0.5
        q3 = math.cos(t * 0.05) * 0.8
        q4 = math.sin(t * 0.05)
        q5 = math.cos(t * 0.05) * 0.5
        q6 = math.cos(t * 0.2) * 0.8

        scene.set_joint_angles({
            "shoulder_pan_joint": q1,
            "shoulder_lift_joint": q2,
            "elbow_joint": q3,
            "wrist_1_joint": q4,
            "wrist_2_joint": q5,
            "wrist_3_joint": q6
        })
        
        # Print progress every 10 steps
        if t % 10 == 0:
            ee_pos = end_effector_pose[t + 1][:3]
            print(f"Step {t}: EE position = [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}], status = {status}")
    
    print("\nSimulation complete!")
    print(f"Final end-effector position: {end_effector_pose[-1][:3]}")

  

if __name__ == "__main__":

    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    quadratic_surface = 2*x**2 + 2*y**2 + 2*x*y + 2*x + 2*y + 2

    q_0 = np.array([0,0,0,0,0,0], dtype=np.float64) #Initial angles
    qdot_0 = np.array([0,0,0,0,0,0], dtype=np.float64) #Initial angular speeds
    robot_loader = urdf("ur5")
    scene = robot_visualizer(robot_loader)
    
    robot = six_dof_model(
        urdf_loader=robot_loader,
        initial_state = np.hstack((q_0, qdot_0)),
        integration_method="RK2"
    )

    mpc = model_predictive_control(
        surface=quadratic_surface,
        state=robot.state,
        control_input=robot.control,
        dynamics=robot.get_explicit_model()['ode'],
        forward_kinematics=robot.fk_casadi,
        differential_kinematics=robot.dk_casadi,
    )

    scene.add_surface_from_casadi(
        quadratic_surface, x, y,
        x_limits=(-0.5, 0.5),
        y_limits=(-0.3, 0.3),
        resolution=80,
        path="surfaces/quadratic_surface",
        color=0x3399FF,
        opacity=0.6,    
        origin=(-0.5, 1.5, 0.2),             # set position here
        orientation_rpy=(0.9, 0.0, 0.4),    # optional roll, pitch, yaw (rad)
    )

    run_sim(scene, robot, mpc.solver, total_time = 100)