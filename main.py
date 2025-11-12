import casadi as ca
import numpy as np

from acados import MPC as model_predictive_control
from loader import UrdfLoader as urdf
# from model import SixDofRobot as six_dof_model
from model_casadi import SixDofRobot as six_dof_model


def run_sim(model, solver, total_time):

    time = np.linspace(0, total_time, num=total_time)
    q = np.zeros((total_time, model.n_dof)) # joint positions
    q_dot = np.zeros((total_time, model.n_dof)) # joint velocities

    end_effector_pose = np.zeros((total_time, 7)) # end-effector pose [x, y, z, qx, qy, qz, qw]
    end_effector_velocity = np.zeros((total_time, 6)) # end-effector velocity [vx, vy, vz, wx, wy, wz]

#Control Loop
    for t in range(total_time - 1):

        current_q = q[t]
        current_q_dot = q_dot[t]
        current_end_effector_pose = end_effector_pose[t]
        current_end_effector_velocity = end_effector_velocity[t]
        
        # Set initial state constraint: [q, q_dot]
        current_state = np.concatenate((current_q, current_q_dot))
        solver.set(0, 'lbx', current_state)
        solver.set(0, 'ubx', current_state)

        solver.solve()
        optimal_control = solver.get(0, "u")

        q_dot[t + 1] = optimal_control

        q[t + 1] = model.update(q[t], optimal_control)
        end_effector_pose[t + 1] = model.forward_kinematics(q[t + 1])
        end_effector_velocity[t + 1] = model.differential_kinematics(q[t + 1], q_dot[t + 1])

  

if __name__ == "__main__":

    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    quadratic_surface = 2*x**2 + 2*y**2 + 2*x*y + 2*x + 2*y + 2

    q_0 = np.array([0,0,0,0,0,0], dtype=np.float64) #Initial angles
    qdot_0 = np.array([0,0,0,0,0,0], dtype=np.float64) #Initial angular speeds
    robot_loader = urdf("ur5")
    
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
    run_sim(robot, total_time = 100)