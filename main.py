import casadi as ca
import numpy as np

from acados import Acados as mpc
from loader import UrdfLoader as urdf
from model import SixDofRobot as six_dof_model




def run_sim(model, total_time):
    time = np.linspace(0, total_time, num=total_time)
    for t in range(total_time):
        q1, q2, q3, q4, q5, q6, = model.update(t)

        # solver.set(0, 'lbx', x_current)
        # solver.set(0, 'ubx', x_current)
        
        # solver.solve()
        # u_opt = solver.get(0, "u")

  

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
        integration_method="RK4"
    )

    # controller = mpc(
    #     surface = surface,
    #     state = model.state,
    #     control_input = model.control,
    #     explicit_model= model.get_explicit_model(),
    #     implisit_model= model.get_implcit_model()
    # )

    run_sim(robot_model, total_time = 100)