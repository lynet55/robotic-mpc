from acados import Acados as mpc
from loader import UrdfLoader as robotModel
import casadi as ca


def run_sim( n_timesteps):


    for i in range(n_timesteps):
        print("hei")

        # solver.set(0, 'lbx', x_current)
        # solver.set(0, 'ubx', x_current)
        
        # solver.solve()
        # u_opt = solver.get(0, "u")

  

if __name__ == "__main__":
    
    robot_model = robotModel("ur5")

    #replace with surface class
    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    
    #####

    # controller = mpc(
    #     surface = surface,
    #     state = model.state,
    #     control_input = model.control,
    #     explicit_model= model.get_explicit_model(),
    #     implisit_model= model.get_implcit_model()
    # )

    #make yaml loader
    print("ss")
    run_sim(n_timesteps = 100)