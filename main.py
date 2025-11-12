import casadi as ca
import numpy as np

from acados import Acados as mpc
from loader import UrdfLoader as urdf
# from model import SixDofRobot as six_dof_model
from model_casadi import SixDofRobot as six_dof_model


def run_sim(model, total_time):

    time = np.linspace(0, total_time, num=total_time)
    state = np.zeros((total_time, model.state_dimension))
    state[0] = model._z0

    for t in range(total_time - 1):

        current_state = state[t]

        # solver.set(0, 'lbx', current_state)
        # solver.set(0, 'ubx', current_state)
        
        # solver.solve()
        # optimal_control = solver.get(0, "u")
    
        optimal_control = np.zeros(6)
        next_state = model.update(current_state, optimal_control)
        state[t + 1] = next_state
        
        q = next_state[:6]  # positions
        q_dot = next_state[6:]  # velocities

        print(f"Time {t}: q = {q}, q_dot = {q_dot}")


  

if __name__ == "__main__":
    #replace with surface class
    x = ca.SX.sym("x")
    y = ca.SX.sym("y")

    q_0 = np.array([0,0,0,0,0,0], dtype=np.float64) #Initial angles
    qdot_0 = np.array([0,0,0,0,0,0], dtype=np.float64) #Initial angular speeds
    robot_loader = urdf("ur5")
    
    robot_model = six_dof_model(
        urdf_loader=robot_loader,
        initial_state = np.hstack((q_0, qdot_0)),
        integration_method="RK2"
    )

    # controller = mpc(
    #     surface = surface,
    #     state = model.state,
    #     control_input = model.control,
    #     explicit_model= model.get_explicit_model(),
    #     implisit_model= model.get_implcit_model()
    # )

    run_sim(robot_model, total_time = 100)