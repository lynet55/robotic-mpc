from acados import Acados as mpc
import casadi as ca


def run_sim(controller, solver, surface, n_timesteps):


    for i in range(n_timesteps):

        solver.set(0, 'lbx', x_current)
        solver.set(0, 'ubx', x_current)
        
        solver.solve()
        u_opt = solver.get(0, "u")

  

if __name__ == "_main_":

    #replace with surface class
    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    surface = 2*x**2 + 2*y**2 + 2*x*y + 2*x + 2*y + 2
    #####

    controller = mpc(
        surface = surface,
        state = model.state,
        control_input = model.control,
        explicit_model= model.get_explicit_model(),
        implisit_model= model.get_implcit_model()
    )

    #make yaml loader
    run_sim(controller, n_timesteps = 100)